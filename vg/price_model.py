"""Henry Hub feed-gas price forecasts 1, 4 and 8 weeks ahead: baselines versus a Conv1D + LSTM.

Walk-forward evaluation: for each test year the models are refit on data whose targets were fully
observed before 1 January of that year, then predict every week of that year.
"""
import numpy as np
import pandas as pd
import torch
from scipy import stats

from vg.contracts import check

HORIZONS = (1, 4, 8)
WINDOW = 52
TEST_YEARS = range(2022, 2027)
SEEDS = (0, 1, 2, 3, 4)
METHODS = {"naive": "Last week's price", "seasonal": "Last week + last year's seasonal change",
           "ridge": "Linear model on recent changes", "conv_lstm": "Conv1D + LSTM"}


def weekly(hh_daily):
    series = hh_daily.set_index("date").henry_hub.clip(lower=0.5).resample("W-FRI").mean().dropna()
    return pd.DataFrame({"week": series.index, "price": series.values, "log_price": np.log(series.values)})


def features(frame):
    """Inputs at each origin: the last 52 weekly log changes and the week-of-year cycle."""
    change = frame.log_price.diff().to_numpy()
    angle = 2 * np.pi * frame.week.dt.isocalendar().week.to_numpy(dtype=float) / 52.18
    inputs = np.stack([change, np.sin(angle), np.cos(angle)], axis=1)
    targets = np.stack([frame.log_price.shift(-h) - frame.log_price for h in HORIZONS], axis=1)
    return inputs, targets


class ConvLSTM(torch.nn.Module):
    def __init__(self, channels=3, filters=16, hidden=32, outputs=len(HORIZONS)):
        super().__init__()
        self.conv = torch.nn.Conv1d(channels, filters, kernel_size=5, padding="same")
        self.lstm = torch.nn.LSTM(filters, hidden, batch_first=True)
        self.head = torch.nn.Linear(hidden, outputs)

    def forward(self, x):                       # x: (batch, time, channels)
        z = torch.relu(self.conv(x.transpose(1, 2))).transpose(1, 2)
        _, (h, _) = self.lstm(z)
        return self.head(h[-1])


def windows(inputs, origins):
    return np.stack([inputs[o - WINDOW + 1:o + 1] for o in origins]).astype(np.float32)


def fit_conv_lstm(x, y, seed, epochs=300, patience=25):
    torch.manual_seed(seed)
    split = int(len(x) * 0.85)                  # last 15% of the training period for early stopping
    model = ConvLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    xt, yt = torch.tensor(x[:split]), torch.tensor(y[:split])
    xv, yv = torch.tensor(x[split:]), torch.tensor(y[split:])
    best, best_state, waited = np.inf, None, 0
    for _ in range(epochs):
        model.train()
        for batch in torch.randperm(len(xt)).split(64):
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(model(xt[batch]), yt[batch])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val = torch.nn.functional.l1_loss(model(xv), yv).item()
        if val < best - 1e-5:
            best, best_state, waited = val, {k: v.clone() for k, v in model.state_dict().items()}, 0
        elif (waited := waited + 1) >= patience:
            break
    model.load_state_dict(best_state)
    return model


def split_origins(weeks, year):
    """Training origins need every horizon's target observed before 1 January of the test year."""
    start, max_h = pd.Timestamp(f"{year}-01-01"), max(HORIZONS)
    train = np.array([o for o in range(WINDOW + 1, len(weeks) - max_h) if weeks.iloc[o + max_h] < start])
    test = np.array([o for o in range(WINDOW + 1, len(weeks)) if weeks.iloc[o].year == year])
    return train, test


def predict_year(frame, inputs, targets, year):
    """Train on origins whose 8-week target is observed before `year`, predict origins within `year`."""
    train, test = split_origins(frame.week, year)
    if not len(test):
        return None
    mean, std = inputs[train, 0].mean(), inputs[train, 0].std()
    scaled = inputs.copy()
    scaled[:, 0] = (np.nan_to_num(scaled[:, 0]) - mean) / std
    y_scale = targets[train].std(axis=0)
    out = {"naive": np.zeros((len(test), len(HORIZONS)))}
    out["seasonal"] = np.stack([frame.log_price.shift(52 - h).to_numpy()[test] - frame.log_price.shift(52).to_numpy()[test]
                                for h in HORIZONS], axis=1)
    lags = lambda origins: np.column_stack([scaled[origins - k, 0] for k in range(12)] + [scaled[origins, 1], scaled[origins, 2],
                                            np.ones(len(origins))])
    a, lam = lags(train), 5.0
    coef = np.linalg.solve(a.T @ a + lam * np.eye(a.shape[1]), a.T @ targets[train])
    out["ridge"] = lags(test) @ coef
    xw, xtest = windows(scaled, train), windows(scaled, test)
    ensemble = []
    for seed in SEEDS:
        model = fit_conv_lstm(xw, (targets[train] / y_scale).astype(np.float32), seed)
        with torch.no_grad():
            ensemble.append(model(torch.tensor(xtest)).numpy() * y_scale)
    out["conv_lstm"] = np.mean(ensemble, axis=0)
    rows = []
    for i, origin in enumerate(test):
        for j, h in enumerate(HORIZONS):
            known = origin + h < len(frame)
            if not known and origin != len(frame) - 1:
                continue
            row = {"origin": frame.week.iloc[origin], "target_week": frame.week.iloc[origin] + pd.Timedelta(weeks=h),
                   "horizon_weeks": h, "price_at_origin": frame.price.iloc[origin],
                   "actual": frame.price.iloc[origin + h] if known else np.nan}
            for method, values in out.items():
                row[method] = float(np.exp(frame.log_price.iloc[origin] + np.nan_to_num(values[i, j])))
            rows.append(row)
    return pd.DataFrame(rows)


def diebold_mariano(errors_a, errors_b, h):
    """Two-sided DM test on absolute errors with a Newey-West variance (h-1 lags)."""
    d = np.abs(errors_a) - np.abs(errors_b)
    n, mean = len(d), d.mean()
    gamma = [np.mean((d[k:] - mean) * (d[:n - k] - mean)) for k in range(h)]
    variance = (gamma[0] + 2 * sum((1 - k / h) * gamma[k] for k in range(1, h))) / n
    stat = mean / np.sqrt(variance)
    return float(stat), float(2 * (1 - stats.norm.cdf(abs(stat))))


def evaluate(predictions):
    rows = []
    for h, group in predictions.groupby("horizon_weeks"):
        naive_mae = (group.naive - group.actual).abs().mean()
        for method in METHODS:
            error = group[method] - group.actual
            stat, p = diebold_mariano(error.to_numpy(), (group.naive - group.actual).to_numpy(), h) if method != "naive" else (0.0, 1.0)
            rows.append({"horizon_weeks": h, "method": method, "label": METHODS[method], "mae": float(error.abs().mean()),
                         "skill_vs_naive": float(1 - error.abs().mean() / naive_mae), "dm_stat": stat, "dm_p_value": p,
                         "n": len(group)})
    return pd.DataFrame(rows)


def analyze(hh_daily):
    torch.set_num_threads(2)
    frame = weekly(hh_daily)
    inputs, targets = features(frame)
    everything = pd.concat([p for year in TEST_YEARS if (p := predict_year(frame, inputs, targets, year)) is not None],
                           ignore_index=True)
    outlook = everything[everything.actual.isna()].reset_index(drop=True)
    predictions = check(everything.dropna(subset=["actual"]).reset_index(drop=True), "gold.henry_hub_forecasts", ["origin", "horizon_weeks", "actual", *METHODS],
                        unique=["origin", "horizon_weeks"], ranges={"actual": (0.5, 60)})
    metrics = evaluate(predictions)
    return frame, predictions, metrics, outlook
