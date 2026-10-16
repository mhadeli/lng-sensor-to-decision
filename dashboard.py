"""Build a one-page dashboard from the pipeline outputs into site/index.html."""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import recall_score

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
SILVER = ROOT / "data" / "silver"
GOLD = ROOT / "data" / "gold"
SITE = ROOT / "site"

BLUE, ORANGE, GRAY, LIGHT_GRAY, INK = "#2a78d6", "#eb6834", "#8a8984", "#c9c8c2", "#0b0b0b"
VENTURE_GLOBAL = ["Calcasieu Pass", "Plaquemines"]


def style(figure, title, y_title=None, percent=False):
    figure.update_layout(
        title=dict(text=title, x=0, font=dict(size=15, color=INK)),
        template="plotly_white",
        height=360,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=-0.15),
        hovermode="x unified",
        font=dict(family="system-ui, sans-serif", color="#52514e"),
    )
    figure.update_yaxes(title=y_title, tickformat=".0%" if percent else ",")
    return figure


def export_charts():
    exports = pd.read_csv(SILVER / "exports.csv", parse_dates=["month"]).set_index("month")
    gulf = pd.read_csv(RAW / "gulf_lng_exports.csv", parse_dates=["month"])
    terminals = gulf.pivot_table(index="month", columns="terminal", values="exports_mmcf", aggfunc="sum")
    volume = exports.join(terminals).fillna(0).loc["2016":] / 1000
    venture_global = volume[VENTURE_GLOBAL].sum(axis=1)
    other_gulf = volume[terminals.columns].sum(axis=1) - venture_global
    rest = (volume["exports_mmcf"] - venture_global - other_gulf).clip(lower=0)

    stack = go.Figure()
    for name, values, color in [
        ("Venture Global", venture_global, BLUE),
        ("Other Gulf terminals", other_gulf, GRAY),
        ("Rest of U.S.", rest, LIGHT_GRAY),
    ]:
        stack.add_scatter(x=volume.index, y=values, name=name, stackgroup="one", line=dict(width=0.5, color=color), fillcolor=color)
    style(stack, "U.S. LNG exports by source", "Bcf per month")

    # Exports against their centered 12-month average, by calendar month.
    total = volume["exports_mmcf"]
    ratio = (total / total.rolling(12, center=True).mean()).loc["2019":].dropna() - 1
    seasonal = ratio.groupby(ratio.index.month).mean()
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    season = go.Figure(go.Bar(
        x=months, y=seasonal, marker_color=[ORANGE if value < 0 else BLUE for value in seasonal],
        hovertemplate="%{x}: %{y:+.1%}<extra></extra>",
    ))
    style(season, "Exports against their 12-month trend, by calendar month (2019 onward)", percent=True)

    last_year = volume.index >= volume.index.max() - pd.DateOffset(months=11)
    share_last_year = venture_global[last_year].sum() / total[last_year].sum()
    return stack, season, share_last_year, volume.index.max()


def forecast_charts():
    forecast = pd.read_csv(GOLD / "volume_forecast.csv", parse_dates=["month"])
    lines = go.Figure()
    lines.add_scatter(x=forecast["month"], y=forecast["actual"] / 1000, name="Actual", line=dict(color=INK, width=3))
    lines.add_scatter(x=forecast["month"], y=forecast["forecast"] / 1000, name="Forecast", line=dict(color=BLUE))
    lines.add_scatter(
        x=forecast["month"], y=forecast["same_as_last_month"] / 1000, name="Same as last month",
        line=dict(color=ORANGE, dash="dot"),
    )
    style(lines, "Next-month exports: actual against forecast", "Bcf")

    methods = {
        "same_as_last_month": "Same as last month (baseline)",
        "same_daily_rate": "Last month's daily rate × days in the month",
        "calendar_month": "Daily rate × usual calendar-month change",
        "forecast": "Average of the two (final)",
    }
    miss = forecast[list(methods)].sub(forecast["actual"], axis=0).abs().mean() / 1000
    table = pd.DataFrame({
        "Method": [methods[key] for key in methods],
        "Average miss (Bcf)": miss.round(1).to_numpy(),
        "vs baseline": [f"{value / miss.iloc[0] - 1:+.0%}" if i else "" for i, value in enumerate(miss)],
    })
    return lines, table, miss, len(forecast)


def fault_charts():
    predictions = pd.read_csv(GOLD / "fault_predictions.csv")
    caught = predictions[predictions["is_fault"] == 1].groupby("fault_id")[["rule_predicted", "model_predicted"]].mean()
    dots = go.Figure()
    for column, name, color in [("rule_predicted", "Rule", GRAY), ("model_predicted", "Random forest", BLUE)]:
        dots.add_scatter(x=caught.index, y=caught[column], name=name, mode="markers", marker=dict(size=10, color=color))
    false_alarms = predictions.loc[predictions["is_fault"] == 0, "model_predicted"].mean()
    dots.add_hline(y=false_alarms, line=dict(color=ORANGE, dash="dash", width=1),
                   annotation_text="random forest false-alarm rate", annotation_position="bottom right")
    style(dots, "Share of fault rows caught, by fault", percent=True)
    dots.update_xaxes(title="fault", dtick=1)
    dots.update_yaxes(range=[0, 1.05])
    dots.update_layout(hovermode="closest")

    scores = {
        name: (recall_score(predictions["is_fault"], predictions[column]),
               recall_score(predictions["is_fault"], predictions[column], pos_label=0))
        for name, column in [("Rule", "rule_predicted"), ("Random forest", "model_predicted")]
    }
    actual = predictions["is_fault"].map({0: "Actual normal", 1: "Actual fault"})
    confusion = pd.concat(
        {
            name: pd.crosstab(actual, predictions[column].map({0: "said normal", 1: "said fault"}), rownames=[None], colnames=[None])
            .loc[["Actual normal", "Actual fault"], ["said normal", "said fault"]]
            for name, column in [("Rule", "rule_predicted"), ("Random forest", "model_predicted")]
        },
        axis=1,
    )
    return dots, scores, confusion


def chart(figure, first=False):
    return figure.to_html(full_html=False, include_plotlyjs="cdn" if first else False, config={"displayModeBar": False, "responsive": True})


def main():
    stack, season, share, latest = export_charts()
    lines, methods, miss, test_months = forecast_charts()
    dots, scores, confusion = fault_charts()

    tiles = [
        (f"{share:.0%}", "Venture Global share of U.S. LNG exports, last 12 months"),
        (f"{miss.iloc[-1]:.1f} Bcf", f"average forecast miss, against {miss.iloc[0]:.1f} for the baseline"),
        (f"{scores['Random forest'][0]:.0%}", f"fault rows caught by the model ({scores['Rule'][0]:.0%} by the rule)"),
        (f"{scores['Random forest'][1]:.0%}", f"normal rows cleared by the model ({scores['Rule'][1]:.0%} by the rule)"),
    ]
    tile_html = "".join(f'<div class="tile"><b>{value}</b><span>{label}</span></div>' for value, label in tiles)

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LNG Sensor to Decision</title>
<style>
  :root {{ --ink: #0b0b0b; --muted: #52514e; --line: #e5e4e0; --surface: #fcfcfb; --card: #ffffff; }}
  body {{ margin: 0; background: var(--surface); color: var(--ink); font: 15px/1.55 system-ui, sans-serif; }}
  main {{ max-width: 1040px; margin: 0 auto; padding: 32px 16px 64px; }}
  h1 {{ margin: 0 0 4px; font-size: 28px; }}
  h2 {{ margin: 40px 0 8px; font-size: 20px; }}
  p {{ color: var(--muted); max-width: 760px; }}
  .tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin: 24px 0; }}
  .tile {{ background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 16px; }}
  .tile b {{ display: block; font-size: 28px; }}
  .tile span {{ color: var(--muted); font-size: 13px; }}
  .card .plotly-graph-div {{ width: 100% !important; }}
  h1 {{ overflow-wrap: anywhere; }}
  .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 12px; margin: 12px 0; overflow-x: auto; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
  th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line); }}
  footer {{ margin-top: 48px; color: var(--muted); font-size: 13px; }}
</style>
</head>
<body>
<main>
<h1>LNG portfolio: sensor to decision</h1>
<p>Public stand-ins for plant data, from sensor alerts to a monthly export plan. Rebuilt automatically from the source data every month. Latest export month: {latest:%B %Y}.</p>
<div class="tiles">{tile_html}</div>

<h2>Where U.S. LNG comes from</h2>
<p>Venture Global's Calcasieu Pass and Plaquemines now ship about {share:.0%} of U.S. LNG. Exports dip every summer, when hot air cuts liquefaction capacity.</p>
<div class="card">{chart(stack, first=True)}</div>
<div class="card">{chart(season)}</div>

<h2>Next-month export forecast</h2>
<p>Each of the last {test_months} months is forecast from earlier months only and compared with the actual value. The forecast uses the daily export rate and the usual change into each calendar month.</p>
<div class="card">{chart(lines)}</div>
<div class="card">{methods.to_html(index=False, border=0)}</div>

<h2>Fault detection</h2>
<p>Tennessee Eastman process runs stand in for plant historian tags. Both methods are scored on test runs the model never trained on. Faults 3, 9, and 15 barely move the sensors and are not detected by either.</p>
<div class="card">{chart(dots)}</div>
<div class="card">{confusion.to_html(border=0)}</div>

<footer>Data: EIA, NOAA, Tennessee Eastman benchmark (Braatz), Microsoft predictive-maintenance sample. Built by <code>dashboard.py</code>.</footer>
</main>
</body>
</html>
"""
    SITE.mkdir(exist_ok=True)
    (SITE / "index.html").write_text(html)
    print(f"wrote {(SITE / 'index.html').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
