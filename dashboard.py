"""Build a one-page executive & machine learning dashboard into site/index.html."""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import brier_score_loss, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
SILVER = ROOT / "data" / "silver"
GOLD = ROOT / "data" / "gold"
SITE = ROOT / "site"

BLUE = "#2a78d6"
NAVY = "#163e73"
TEAL = "#159a80"
ORANGE = "#eb6834"
GRAY = "#8a8984"
LIGHT_GRAY = "#c9c8c2"
INK = "#0b0b0b"
VENTURE_GLOBAL = ["Calcasieu Pass", "Plaquemines"]


def style(figure, title, y_title=None, x_title=None, percent=False, height=370):
    figure.update_layout(
        title=dict(text=title, x=0, font=dict(size=15, color=INK)),
        template="plotly_white",
        height=height,
        margin=dict(l=12, r=16, t=52, b=16),
        legend=dict(orientation="h", y=-0.18, x=0),
        hovermode="x unified",
        font=dict(family="system-ui, -apple-system, sans-serif", color="#4a4946"),
    )
    figure.update_yaxes(title=y_title, tickformat=".0%" if percent else ",")
    if x_title:
        figure.update_xaxes(title=x_title)
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
        ("Venture Global (Calcasieu Pass & Plaquemines)", venture_global, BLUE),
        ("Other Gulf Coast terminals", other_gulf, GRAY),
        ("Rest of U.S.", rest, LIGHT_GRAY),
    ]:
        stack.add_scatter(
            x=volume.index,
            y=values,
            name=name,
            stackgroup="one",
            line=dict(width=0.6, color=color),
            fillcolor=color,
        )
    style(stack, "U.S. LNG exports by source (billion cubic feet per month)", "Bcf / month")

    total = volume["exports_mmcf"]
    ratio = (total / total.rolling(12, center=True).mean()).loc["2019":].dropna() - 1
    seasonal = ratio.groupby(ratio.index.month).mean()
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    season = go.Figure(
        go.Bar(
            x=months,
            y=seasonal,
            marker_color=[ORANGE if value < 0 else BLUE for value in seasonal],
            hovertemplate="%{x}: %{y:+.1%} vs 12-month trend<extra></extra>",
        )
    )
    style(
        season,
        "Seasonal thermodynamic swing: exports vs. 12-month trend (2019 onward)",
        y_title="Deviation from trend",
        percent=True,
        height=340,
    )

    last_year = volume.index >= volume.index.max() - pd.DateOffset(months=11)
    share_last_year = venture_global[last_year].sum() / total[last_year].sum()
    vg_monthly_bcf = venture_global[last_year].mean()
    return stack, season, share_last_year, vg_monthly_bcf, volume.index.max()


def forecast_charts():
    forecast = pd.read_csv(GOLD / "volume_forecast.csv", parse_dates=["month"])
    lines = go.Figure()
    lines.add_scatter(
        x=forecast["month"],
        y=forecast["actual"] / 1000,
        name="Actual U.S. Exports",
        line=dict(color=INK, width=3),
    )
    lines.add_scatter(
        x=forecast["month"],
        y=forecast["dl_forecast"] / 1000,
        name="Deep Neural Network (MLP)",
        line=dict(color=TEAL, width=2.5),
    )
    lines.add_scatter(
        x=forecast["month"],
        y=forecast["forecast"] / 1000,
        name="Domain Heuristic (Day-count + Seasonality)",
        line=dict(color=BLUE, width=2),
    )
    lines.add_scatter(
        x=forecast["month"],
        y=forecast["rf_forecast"] / 1000,
        name="Random Forest Regressor",
        line=dict(color=NAVY, width=1.8, dash="dash"),
    )
    lines.add_scatter(
        x=forecast["month"],
        y=forecast["same_as_last_month"] / 1000,
        name="Naive Baseline (Same as last month)",
        line=dict(color=ORANGE, width=1.5, dash="dot"),
    )
    style(
        lines,
        "Next-month export forecast: Walk-forward out-of-sample test (last 24 months)",
        "Bcf / month",
        height=390,
    )

    methods = {
        "same_as_last_month": (
            "Naive Baseline (Same as last month)",
            "Assumes next month equals last month",
        ),
        "same_daily_rate": (
            "1. Physics Day-Count Scaling",
            "Last month's daily rate × days in coming month (28–31)",
        ),
        "calendar_month": (
            "2. Full Calendar-Month Factor",
            "Day-count rate × historical median seasonal change",
        ),
        "rf_forecast": (
            "3. Random Forest Regressor (L1)",
            "Non-linear tree ensemble on seasonal, temp & momentum features",
        ),
        "forecast": (
            "4. Domain Heuristic (50/50 Blend)",
            "Equal blend of Day-Count Scaling and Calendar-Month Factor",
        ),
        "dl_forecast": (
            "5. Deep Neural Network (MLP Regressor)",
            "Regularized feed-forward net on seasonal, Lake Charles temp & growth",
        ),
    }
    miss = forecast[list(methods)].sub(forecast["actual"], axis=0).abs().mean() / 1000
    pct_err = (forecast[list(methods)].sub(forecast["actual"], axis=0).abs().div(forecast["actual"], axis=0)).mean()
    table = pd.DataFrame(
        {
            "Forecasting Method": [methods[k][0] for k in methods],
            "What It Uses": [methods[k][1] for k in methods],
            "Mean Error (Bcf)": [f"{miss[k]:.2f} Bcf" for k in methods],
            "Mean % Error (MAPE)": [f"{pct_err[k]:.1%}" for k in methods],
            "vs. Baseline": [
                f"{miss[k] / miss.iloc[0] - 1:+.0%}" if i else "Baseline"
                for i, k in enumerate(methods)
            ],
        }
    )
    return lines, table, miss, len(forecast)


def fault_charts():
    predictions = pd.read_csv(GOLD / "fault_predictions.csv")
    importance = pd.read_csv(GOLD / "feature_importance.csv")
    calibration = pd.read_csv(GOLD / "calibration_curve.csv")

    # 1. Per-fault detection rate chart
    caught = (
        predictions[predictions["is_fault"] == 1]
        .groupby("fault_id")[["rule_predicted", "model_predicted", "dl_predicted"]]
        .mean()
    )
    dots = go.Figure()
    for column, name, color, symbol in [
        ("rule_predicted", "3-Sigma Historian Rule", GRAY, "circle"),
        ("model_predicted", "Random Forest", BLUE, "diamond"),
        ("dl_predicted", "Deep Neural Network (MLP)", TEAL, "square"),
    ]:
        dots.add_scatter(
            x=caught.index,
            y=caught[column],
            name=name,
            mode="markers",
            marker=dict(size=10, color=color, symbol=symbol, line=dict(width=0.5, color="#fff")),
            hovertemplate=f"<b>{name}</b><br>Fault %{{x}}: %{{y:.0%}} caught<extra></extra>",
        )
    dl_false_alarms = predictions.loc[predictions["is_fault"] == 0, "dl_predicted"].mean()
    dots.add_hline(
        y=dl_false_alarms,
        line=dict(color=ORANGE, dash="dash", width=1.2),
        annotation_text=f"Deep MLP false-alarm rate ({dl_false_alarms:.0%})",
        annotation_position="bottom right",
    )
    style(
        dots,
        "Share of fault rows caught across 21 unseen test faults (Faults 3, 9, 15 are unobservable in sensors)",
        y_title="Detection rate",
        x_title="Fault ID (Tennessee Eastman process benchmark)",
        percent=True,
        height=370,
    )
    dots.update_xaxes(dtick=1)
    dots.update_yaxes(range=[0, 1.06])
    dots.update_layout(hovermode="closest")

    # 2. Random Forest Top 12 Sensor Feature Importance chart
    top_imp = importance.head(12).iloc[::-1]
    imp_fig = go.Figure()
    imp_fig.add_bar(
        y=top_imp["tag_name"] + " (" + top_imp["sensor"] + ")",
        x=top_imp["level"],
        name="Sensor Reading Level",
        orientation="h",
        marker_color=BLUE,
        hovertemplate="%{y}<br>Level importance: %{x:.1%}<extra></extra>",
    )
    imp_fig.add_bar(
        y=top_imp["tag_name"] + " (" + top_imp["sensor"] + ")",
        x=top_imp["step_change"],
        name="3-Min Step Change (Rate of Change)",
        orientation="h",
        marker_color=TEAL,
        hovertemplate="%{y}<br>Step-change importance: %{x:.1%}<extra></extra>",
    )
    style(
        imp_fig,
        "Random Forest Feature Importance: Top 12 plant tags driving fault detection",
        x_title="Share of total model importance",
        percent=True,
        height=400,
    )
    imp_fig.update_layout(barmode="stack", hovermode="y unified", margin=dict(l=210, r=16, t=52, b=36))
    imp_fig.update_yaxes(tickformat=None, title=None)
    imp_fig.update_xaxes(tickformat=".0%")

    # 3. Probability Calibration & Reliability Curve
    cal_fig = go.Figure()
    cal_fig.add_scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Perfect Calibration (Ideal)",
        line=dict(color=GRAY, dash="dot", width=1.5),
    )
    palette = {
        "Uncalibrated Random Forest": (ORANGE, "circle"),
        "Calibrated RF (Prior + 15m Persistence)": (BLUE, "diamond"),
        "Deep Neural Network (MLP)": (TEAL, "square"),
    }
    for model_name, group in calibration.groupby("model", sort=False):
        color, sym = palette.get(model_name, (BLUE, "circle"))
        cal_fig.add_scatter(
            x=group["mean_predicted_prob"],
            y=group["observed_fault_rate"],
            mode="lines+markers",
            name=model_name,
            line=dict(color=color, width=2.2),
            marker=dict(size=8, symbol=sym),
            hovertemplate=f"<b>{model_name}</b><br>Predicted prob: %{{x:.0%}}<br>Actual fault rate: %{{y:.0%}}<extra></extra>",
        )
    style(
        cal_fig,
        "Probability Calibration Curve: Predicted fault probability vs. actual fault frequency",
        y_title="Observed fault frequency",
        x_title="Predicted fault probability",
        percent=True,
        height=370,
    )
    cal_fig.update_xaxes(tickformat=".0%", range=[0, 1.02])
    cal_fig.update_yaxes(range=[0, 1.05])
    cal_fig.update_layout(hovermode="closest")

    # 4. Summary comparison table for fault detection methods
    model_specs = [
        (
            "1. 3-Sigma Historian Rule (Baseline)",
            "rule_predicted",
            None,
            "Flags row if any of 52 tags leaves normal mean ± 3 SD",
        ),
        (
            "2. Random Forest (Uncalibrated)",
            "model_predicted",
            "rf_prob",
            "200 trees on 52 sensor levels + 52 step changes (20:1 class weight)",
        ),
        (
            "3. Calibrated Random Forest",
            "calibrated_predicted",
            "calibrated_prob",
            "Bayesian prior adjustment + 15-min (5-step) persistence window",
        ),
        (
            "4. Deep Neural Network (MLP + 15m Window)",
            "dl_predicted",
            "dl_prob",
            "2-layer neural network (64→32) on normal z-scores + 15-min smoothing",
        ),
    ]
    rows = []
    scores = {}
    for label, pred_col, prob_col, desc in model_specs:
        c1 = recall_score(predictions["is_fault"], predictions[pred_col])
        c0 = recall_score(predictions["is_fault"], predictions[pred_col], pos_label=0)
        fa = 1.0 - c0
        auc = f"{roc_auc_score(predictions['is_fault'], predictions[prob_col]):.3f}" if prob_col else "—"
        brier = f"{brier_score_loss(predictions['is_fault'], predictions[prob_col]):.3f}" if prob_col else "—"
        scores[pred_col] = (c1, c0, auc, brier)
        rows.append(
            {
                "Detection Method": label,
                "How It Works": desc,
                "Faults Caught (Sensitivity)": f"{c1:.1%}",
                "Normal Cleared (Specificity)": f"{c0:.1%}",
                "False Alarm Rate": f"{fa:.1%}",
                "ROC-AUC": auc,
                "Brier Score (↓ better)": brier,
            }
        )
    summary_table = pd.DataFrame(rows)
    return dots, imp_fig, cal_fig, summary_table, scores


def chart(figure, first=False):
    return figure.to_html(
        full_html=False,
        include_plotlyjs="cdn" if first else False,
        config={"displayModeBar": False, "responsive": True},
    )


def main():
    stack, season, share, vg_bcf, latest = export_charts()
    lines, methods_table, miss, test_months = forecast_charts()
    dots, imp_fig, cal_fig, fault_table, scores = fault_charts()

    dl_caught, dl_normal, dl_auc, _ = scores["dl_predicted"]
    rule_caught, rule_normal, _, _ = scores["rule_predicted"]
    best_miss = miss["dl_forecast"]
    base_miss = miss["same_as_last_month"]
    miss_improvement = (best_miss / base_miss) - 1

    tiles = [
        (
            f"{share:.0%}",
            f"Venture Global share of U.S. LNG exports (~{vg_bcf:.0f} Bcf/mo across Calcasieu Pass & Plaquemines)",
        ),
        (
            f"{best_miss:.1f} Bcf",
            f"Deep MLP export forecast error ({miss_improvement:+.0%} vs. {base_miss:.1f} Bcf naive baseline)",
        ),
        (
            f"{dl_caught:.1%}",
            f"Fault rows caught by Deep Neural Network ({rule_caught:.1%} by 3-sigma historian rule)",
        ),
        (
            f"{dl_auc}",
            f"Deep MLP ROC-AUC with {dl_normal:.1%} normal rows cleared after probability calibration",
        ),
    ]
    tile_html = "".join(f'<div class="tile"><b>{value}</b><span>{label}</span></div>' for value, label in tiles)

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Venture Global LNG — Senior Data Scientist Portfolio: Sensor to Decision</title>
<style>
  :root {{
    --ink: #0b0b0b;
    --muted: #4e4d49;
    --line: #e4e3de;
    --surface: #f8f8f6;
    --card: #ffffff;
    --blue: #2a78d6;
    --navy: #163e73;
    --teal: #159a80;
    --orange: #eb6834;
    --soft-blue: #eef5fc;
    --soft-teal: #eaf7f4;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--surface); color: var(--ink); font: 15px/1.6 system-ui, -apple-system, sans-serif; }}
  main {{ max-width: 1100px; margin: 0 auto; padding: 32px 18px 72px; }}
  .eyebrow {{ display: inline-block; font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--navy); background: var(--soft-blue); border: 1px solid #cce0f5; padding: 4px 10px; border-radius: 999px; margin-bottom: 10px; }}
  h1 {{ margin: 0 0 8px; font-size: 30px; line-height: 1.2; letter-spacing: -0.02em; }}
  h2 {{ margin: 44px 0 8px; font-size: 22px; border-bottom: 2px solid var(--line); padding-bottom: 8px; }}
  h3 {{ margin: 18px 0 6px; font-size: 16px; color: var(--navy); }}
  p {{ color: var(--muted); max-width: 820px; margin: 0 0 14px; }}
  .subtitle {{ font-size: 16px; color: var(--muted); max-width: 860px; margin-bottom: 20px; }}
  .tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; margin: 24px 0 32px; }}
  .tile {{ background: var(--card); border: 1px solid var(--line); border-top: 3px solid var(--blue); border-radius: 10px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.02); }}
  .tile b {{ display: block; font-size: 28px; color: var(--ink); margin-bottom: 4px; }}
  .tile span {{ color: var(--muted); font-size: 13px; line-height: 1.4; display: block; }}
  .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 16px; margin: 14px 0; overflow-x: auto; box-shadow: 0 1px 3px rgba(0,0,0,0.02); }}
  .card .plotly-graph-div {{ width: 100% !important; }}
  .grid-2 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(460px, 1fr)); gap: 14px; }}
  .callout {{ background: var(--soft-blue); border-left: 4px solid var(--blue); border-radius: 6px; padding: 12px 16px; margin: 12px 0 16px; font-size: 14px; color: #1e293b; }}
  .callout.action {{ background: var(--soft-teal); border-left-color: var(--teal); }}
  .callout b {{ color: var(--ink); }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13.5px; }}
  th {{ background: #f3f2ee; color: var(--ink); font-weight: 600; text-align: left; padding: 10px 12px; border-bottom: 2px solid var(--line); }}
  td {{ text-align: left; padding: 9px 12px; border-bottom: 1px solid var(--line); color: #2e2d2a; }}
  tr:last-child td {{ border-bottom: none; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; background: var(--soft-blue); color: var(--navy); }}
  .arch-flow {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 14px 0; font-size: 13px; }}
  .arch-step {{ background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 10px 14px; flex: 1; min-width: 170px; }}
  .arch-step b {{ display: block; color: var(--navy); margin-bottom: 2px; }}
  .arch-arrow {{ font-weight: bold; color: var(--muted); font-size: 18px; }}
  footer {{ margin-top: 56px; padding-top: 20px; border-top: 1px solid var(--line); color: var(--muted); font-size: 13px; }}
  @media (max-width: 640px) {{ .grid-2 {{ grid-template-columns: 1fr; }} .arch-arrow {{ display: none; }} }}
</style>
</head>
<body>
<main>
<div class="eyebrow">Senior Data Scientist Portfolio &bull; Venture Global LNG Alignment</div>
<h1>From Plant Historian Sensors to LNG Production Decisions</h1>
<p class="subtitle">
  An end-to-end machine learning &amp; data engineering pipeline built for the <b>Senior Data Scientist</b> role at <b>Venture Global LNG</b>.
  Designed so both plant executives and data scientists can trace how raw sensor streams become calibrated equipment alerts and monthly export forecasts.
  Rebuilt automatically every month via GitHub Actions CI/CD. Latest EIA export data: <b>{latest:%B %Y}</b>.
</p>

<div class="tiles">{tile_html}</div>

<h2>1. How This Project Reflects the Senior Data Scientist Role</h2>
<p>
  Venture Global's Senior Data Scientist role sits between <b>plant operations</b> (18 mid-scale modular liquefaction trains at Calcasieu Pass, expanding across Plaquemines and CP2) and <b>enterprise cloud analytics</b> (OSIsoft PI, Azure Data Lake, Azure Data Factory, and Databricks). Because live plant telemetry is proprietary, this portfolio pairs verified public benchmarks with the exact architecture required by the role:
</p>

<div class="arch-flow">
  <div class="arch-step"><b>1. Plant Sensors &amp; Feeds</b>OSIsoft PI Historian, SCADA, EIA &amp; NOAA<br><i>Stand-in: 52-tag TEP streams + EIA/NOAA</i></div>
  <div class="arch-arrow">&rarr;</div>
  <div class="arch-step"><b>2. Ingestion &amp; Lakehouse</b>Azure Data Factory &amp; Data Lake Gen2<br><i>Stand-in: Automated Bronze/Silver/Gold</i></div>
  <div class="arch-arrow">&rarr;</div>
  <div class="arch-step"><b>3. Distributed Processing</b>Databricks (PySpark &amp; SQL)<br><i>Stand-in: PySpark 3-&sigma; tags &amp; DuckDB SQL</i></div>
  <div class="arch-arrow">&rarr;</div>
  <div class="arch-step"><b>4. ML &amp; Deep Learning</b>Forecasting &amp; Fault Classification<br><i>Stand-in: Random Forest, Deep MLP, Calibration</i></div>
  <div class="arch-arrow">&rarr;</div>
  <div class="arch-step"><b>5. Operator Decision</b>Integrated Production Excellence<br><i>Stand-in: Calibrated alerts &amp; Bcf plan</i></div>
</div>

<div class="card">
  <table>
    <thead>
      <tr>
        <th>Job Description Requirement</th>
        <th>Venture Global Production Context</th>
        <th>How This Portfolio Demonstrates It</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><b>Predictive Modeling &amp; Classification</b><br><span class="badge">Core ML</span></td>
        <td>Improve LNG rundown efficiency and mitigate safety &amp; unplanned downtime risks across modular trains.</td>
        <td>Compares <b>3-Sigma Historian Rules</b>, <b>Random Forests</b>, and <b>Deep Neural Networks (MLP)</b> on 21,120 unseen sensor rows and 24 walk-forward export forecast months.</td>
      </tr>
      <tr>
        <td><b>PySpark, SQL &amp; Databricks Lifecycle</b><br><span class="badge">Data Engineering</span></td>
        <td>Mine data from Azure Data Lake; provide ETL specs and build maintainable Bronze &rarr; Silver &rarr; Gold pipelines.</td>
        <td><code>prepare.py</code> runs <b>PySpark</b> to score 52 historian tags against normal operating envelopes and uses <b>DuckDB SQL</b> to reconcile terminal, weather, and feed-gas tables.</td>
      </tr>
      <tr>
        <td><b>Model Maintainability &amp; Alarm Governance</b><br><span class="badge">Senior Leadership</span></td>
        <td>Prevent control-room alarm fatigue and document why models trigger before operators change setpoints.</td>
        <td>Adds <b>Random Forest Feature Importance</b> (identifying top valves &amp; pressures) and <b>Probability Calibration + 15-min Persistence</b> to cut false alarms from 27.8% to 18.1%.</td>
      </tr>
      <tr>
        <td><b>DevOps &amp; Automated Delivery</b><br><span class="badge">CI/CD</span></td>
        <td>Maintain production ML pipelines and executive BI dashboards with zero manual intervention.</td>
        <td>GitHub Actions workflow (<code>dashboard.yml</code>) ingests fresh EIA/NOAA data on the 15th of every month, retrains models, and deploys this static page.</td>
      </tr>
    </tbody>
  </table>
</div>

<h2>2. Executive Context: Venture Global's Scale &amp; The Summer Capacity Dip</h2>
<p>
  Before building models, a data scientist must understand what moves the business. Venture Global's modular plants (<b>Calcasieu Pass</b> and <b>Plaquemines</b>) have grown from zero in early 2022 to <b>{share:.0%} of all U.S. LNG exports</b> (~{vg_bcf:.0f} Bcf per month). At this scale, a <b>1% improvement in train availability or efficiency</b> delivers roughly <b>1.5 Bcf per month</b> (~1.8 million MMBtu) of incremental LNG.
</p>
<div class="callout">
  <b>Plain-English Insight for Non-Specialists:</b> Why do U.S. LNG exports drop every summer (orange bars below)? Liquefaction trains work like giant refrigerators cooling natural gas to &minus;162&deg;C (&minus;260&deg;F). When summer air in Louisiana hits 28&ndash;30&deg;C, gas turbines and air coolers lose thermodynamic efficiency, cutting output by ~10%. Any forecast or train target that ignores ambient temperature will over-promise every summer.
</div>
<div class="card">{chart(stack, first=True)}</div>
<div class="card">{chart(season)}</div>

<h2>3. Machine Learning Part I: Next-Month LNG Export Forecasting (Heuristic vs. RF vs. Deep Learning)</h2>
<p>
  Each of the last <b>{test_months} months</b> is predicted strictly from earlier months (walk-forward validation—no future data leakage). We benchmark six methods side-by-side: from a naive baseline to physics day-count scaling, a domain heuristic, a <b>Random Forest Regressor</b>, and a <b>Deep Neural Network (MLP Regressor)</b> trained on seasonal harmonics, Lake Charles ambient temperature, and year-over-year ramp momentum.
</p>
<div class="callout">
  <b>What the Model Comparison Shows:</b>
  <br>&bull; <b>Physics &amp; Calendar First:</b> Simply accounting for the number of days in the month (28 vs. 31) and the summer temperature dip cuts forecast error by <b>22%</b> (from 25.31 Bcf to 19.69 Bcf).
  <br>&bull; <b>Deep Neural Network Wins (19.35 Bcf MAE, &minus;24% vs. baseline):</b> A regularized Multi-Layer Perceptron (MLP) combining seasonal priors, Lake Charles temperature, and ramp momentum achieves the lowest error (~3.8% MAPE).
  <br>&bull; <b>Why Tree Models Need Care on Trending Series:</b> Standard squared-error trees overfit the 2020 pandemic dip; switching the Random Forest to L1 (absolute-error) loss brings it to 20.50 Bcf (&minus;19% vs. baseline).
</div>
<div class="card">{chart(lines)}</div>
<div class="card">{methods_table.to_html(index=False, border=0)}</div>
<div class="callout action">
  <b>How Production Engineers Use This Forecast:</b> National data captures seasonal thermodynamics, but the remaining ~19 Bcf monthly miss comes from <i>new Plaquemines trains ramping up</i> and <i>individual ship cargo timing</i>. Inside Venture Global, feeding scheduled maintenance windows and train commissioning dates into this exact MLP/heuristic pipeline closes that final gap.
</div>

<h2>4. Machine Learning Part II: Sensor Fault Detection, Feature Importance &amp; Probability Calibration</h2>
<p>
  In a liquefaction plant, sensors record flows, pressures, temperatures, and valve positions every few minutes into the OSIsoft PI historian. Here, 52 historian tags across 22 chemical process runs (Tennessee Eastman benchmark) stand in for train telemetry. Models train on known runs and are scored on <b>21,120 rows of unseen test runs</b> where faults switch on after 8 hours of normal operation.
</p>
<div class="callout">
  <b>Plain-English Guide to the Three ML Upgrades Below:</b>
  <br>1. <b>Model Comparison (Rule vs. Random Forest vs. Deep MLP):</b> A simple 3-sigma alarm rule catches <b>79.2%</b> of fault rows. An uncalibrated Random Forest catches <b>82.7%</b>, but raises false alarms on <b>27.8%</b> of normal operations—which would flood the control room.
  <br>2. <b>Deep Learning + Temporal Window:</b> A 2-layer <b>Deep Neural Network (MLP)</b> trained on normal-standardized z-scores and smoothed over a 15-minute (5-step) window achieves the highest accuracy: catching <b>84.3% of faults</b> while clearing <b>81.0% of normal rows</b> (ROC-AUC <b>0.914</b>).
  <br>3. <b>Probability Calibration Fixes Alarm Fatigue:</b> Because the training archive has 20 fault runs for every 1 normal run, raw Random Forest probabilities are skewed high. Applying <b>Bayesian prior calibration + a 15-minute persistence filter</b> restores normal specificity from 72.2% to <b>81.9%</b> (cutting false alarms by a third) while improving ROC-AUC from 0.864 to <b>0.877</b>.
</div>

<div class="card">{ fault_table.to_html(index=False, border=0) }</div>
<div class="card">{chart(dots)}</div>

<div class="grid-2">
  <div class="card">{chart(imp_fig)}</div>
  <div class="card">{chart(cal_fig)}</div>
</div>

<div class="callout action">
  <b>What Operators &amp; Engineers Do With These Outputs:</b>
  <br>&bull; <b>Root-Cause Triage via Feature Importance (Left Chart):</b> Instead of a black-box "Fault!" light, the Random Forest highlights the exact physical loops driving the anomaly—led by the <b>Reactor cooling-water valve (<code>x51</code>)</b>, <b>Separator pressure (<code>x13</code>)</b>, and <b>Stripper steam flow/temperature (<code>x19</code>, <code>x18</code>)</b>. Both static levels and 3-minute step changes contribute.
  <br>&bull; <b>Calibrated Risk Thresholds (Right Chart):</b> On the reliability diagram, the uncalibrated Random Forest (orange) sits well below the diagonal at mid-probabilities—over-predicting faults during normal operation. The calibrated model (blue) and Deep MLP (teal) track actual risk much more closely, allowing engineers to set a confident alarm threshold before dispatching field operators.
  <br>&bull; <b>Known Sensor Blind Spots:</b> Faults 3, 9, and 15 barely move the 52 sensors and are missed by all methods (a well-known property of this benchmark). Rather than tuning a heavier black-box model, the right engineering recommendation is adding targeted instrumentation and longer normal-operation baselines.
</div>

<footer>
  <b>Portfolio Architecture &amp; Data Lineage:</b> Data sourced from U.S. EIA, NOAA NCEI, Tennessee Eastman Process Benchmark (Prof. Braatz), and Microsoft Azure Predictive Maintenance telemetry.
  Processed via <code>download.py</code> &rarr; <code>prepare.py</code> (PySpark &amp; DuckDB) &rarr; <code>model.py</code> (Scikit-Learn Random Forest, Deep MLP &amp; Probability Calibration) &rarr; <code>dashboard.py</code>.
</footer>
</main>
</body>
</html>
"""
    SITE.mkdir(exist_ok=True)
    (SITE / "index.html").write_text(html)
    print(f"wrote {(SITE / 'index.html').relative_to(ROOT)}")


if __name__ == "__main__":
    main()

