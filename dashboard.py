"""Build a one-page dashboard from the pipeline outputs into site/index.html.

The page follows the project's story: the problem, the questions, the data,
the workflow and tools, then the results. Every number on the page is computed
from data/gold, so the monthly rebuild keeps the text and the charts in step.
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import recall_score, roc_auc_score

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
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

FORECAST_METHODS = {
    "same_as_last_month": ("Same as last month (baseline)", "Next month equals last month"),
    "same_daily_rate": ("Day-count scaling", "Last month's daily rate × days in the coming month"),
    "calendar_month": ("Calendar-month factor", "Day-count rate × median change into that calendar month"),
    "forecast": ("Day-count + seasonal blend", "Average of the two methods above"),
    "rf_forecast": ("Random forest regressor", "Trees on seasonal, temperature and momentum features"),
    "dl_forecast": ("Neural network (MLP regressor)", "Small regularized network on the same features"),
}
FAULT_METHODS = [
    ("3-sigma rule (baseline)", "rule_predicted", None, "Alert if any of 52 tags leaves the normal mean ± 3 SD"),
    ("Random forest", "model_predicted", "rf_prob", "200 trees on sensor levels and 3-minute changes"),
    ("Calibrated random forest", "calibrated_predicted", "calibrated_prob",
     "Probability adjusted for the 20:1 training mix, then averaged over 15 minutes"),
    ("Neural network (MLP)", "dl_predicted", "dl_prob", "2-layer network on normal-scaled readings, averaged over 15 minutes"),
]


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


def chart(figure, first=False):
    return figure.to_html(
        full_html=False,
        include_plotlyjs="cdn" if first else False,
        config={"displayModeBar": False, "responsive": True},
    )


def table(frame):
    return frame.to_html(index=False, border=0, escape=False)


def export_results():
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
        stack.add_scatter(x=volume.index, y=values, name=name, stackgroup="one",
                          line=dict(width=0.6, color=color), fillcolor=color)
    style(stack, "U.S. LNG exports by source", "Bcf per month")

    total = volume["exports_mmcf"]
    ratio = (total / total.rolling(12, center=True).mean()).loc["2019":].dropna() - 1
    seasonal = ratio.groupby(ratio.index.month).mean()
    season = go.Figure(go.Bar(
        x=MONTHS, y=seasonal, marker_color=[ORANGE if value < 0 else BLUE for value in seasonal],
        hovertemplate="%{x}: %{y:+.1%} against the 12-month trend<extra></extra>",
    ))
    style(season, "Exports against their 12-month trend, by calendar month (2019 onward)",
          "Difference from trend", percent=True, height=340)

    last_year = volume.index >= volume.index.max() - pd.DateOffset(months=11)
    return {
        "stack": stack,
        "season": season,
        "share": venture_global[last_year].sum() / total[last_year].sum(),
        "vg_bcf": venture_global[last_year].mean(),
        "latest": volume.index.max(),
        "summer": seasonal.loc[6:9].mean(),
        "winter": seasonal.loc[[12, 1, 3]].mean(),
    }


def forecast_results():
    forecast = pd.read_csv(GOLD / "volume_forecast.csv", parse_dates=["month"])
    miss = forecast[list(FORECAST_METHODS)].sub(forecast["actual"], axis=0).abs()
    mean_miss = miss.mean() / 1000
    mean_pct = miss.div(forecast["actual"], axis=0).mean()
    best = mean_miss.drop("same_as_last_month").idxmin()

    lines = go.Figure()
    lines.add_scatter(x=forecast["month"], y=forecast["actual"] / 1000, name="Actual", line=dict(color=INK, width=3))
    for column, color, dash in [
        ("dl_forecast", TEAL, None),
        ("forecast", BLUE, None),
        ("rf_forecast", NAVY, "dash"),
        ("same_as_last_month", ORANGE, "dot"),
    ]:
        lines.add_scatter(
            x=forecast["month"], y=forecast[column] / 1000, name=FORECAST_METHODS[column][0],
            line=dict(color=color, width=2, dash=dash),
            visible=True if column in (best, "same_as_last_month") else "legendonly",
        )
    style(lines, "Next-month exports: actual against forecast (click the legend to add methods)", "Bcf per month", height=390)

    rows = pd.DataFrame({
        "Method": [FORECAST_METHODS[key][0] for key in FORECAST_METHODS],
        "What it uses": [FORECAST_METHODS[key][1] for key in FORECAST_METHODS],
        "Average miss": [f"{mean_miss[key]:.1f} Bcf" for key in FORECAST_METHODS],
        "Average miss, %": [f"{mean_pct[key]:.1%}" for key in FORECAST_METHODS],
        "vs baseline": ["" if key == "same_as_last_month" else f"{mean_miss[key] / mean_miss['same_as_last_month'] - 1:+.0%}"
                        for key in FORECAST_METHODS],
    })
    return {
        "lines": lines,
        "table": rows,
        "miss": mean_miss,
        "pct": mean_pct,
        "best": best,
        "months": len(forecast),
        "first": forecast["month"].min(),
        "last": forecast["month"].max(),
    }


def fault_results():
    predictions = pd.read_csv(GOLD / "fault_predictions.csv")
    importance = pd.read_csv(GOLD / "feature_importance.csv")
    calibration = pd.read_csv(GOLD / "calibration_curve.csv")
    actual = predictions["is_fault"]

    scores = {}
    for name, predicted, probability, how in FAULT_METHODS:
        scores[name] = {
            "caught": recall_score(actual, predictions[predicted]),
            "cleared": recall_score(actual, predictions[predicted], pos_label=0),
            "auc": roc_auc_score(actual, predictions[probability]) if probability else None,
            "how": how,
        }
    # Rank by balanced accuracy: catching faults and clearing normal rows count equally.
    best = max(scores, key=lambda name: scores[name]["caught"] + scores[name]["cleared"])
    rule = FAULT_METHODS[0][0]
    rows = pd.DataFrame({
        "Method": list(scores),
        "How it works": [score["how"] for score in scores.values()],
        "Faults caught": [f"{score['caught']:.1%}" for score in scores.values()],
        "Normal rows cleared": [f"{score['cleared']:.1%}" for score in scores.values()],
        "False alarms": [f"{1 - score['cleared']:.1%}" for score in scores.values()],
        "ROC-AUC": [f"{score['auc']:.3f}" if score["auc"] else "—" for score in scores.values()],
    })

    columns = [predicted for _, predicted, _, _ in FAULT_METHODS]
    caught = predictions[actual == 1].groupby("fault_id")[columns].mean()
    blind_spots = caught.max(axis=1).loc[lambda rate: rate < 0.6].index.tolist()

    dots = go.Figure()
    for (name, predicted, _, _), color, symbol in zip(
        FAULT_METHODS, [GRAY, BLUE, NAVY, TEAL], ["circle", "diamond", "cross", "square"]
    ):
        dots.add_scatter(
            x=caught.index, y=caught[predicted], name=name, mode="markers",
            marker=dict(size=10, color=color, symbol=symbol, line=dict(width=0.5, color="#fff")),
            hovertemplate=f"<b>{name}</b><br>Fault %{{x}}: %{{y:.0%}} caught<extra></extra>",
            visible=True if name in (rule, best) else "legendonly",
        )
    best_column = dict((name, predicted) for name, predicted, _, _ in FAULT_METHODS)[best]
    false_alarms = predictions.loc[actual == 0, best_column].mean()
    dots.add_hline(y=false_alarms, line=dict(color=ORANGE, dash="dash", width=1.2),
                   annotation_text=f"{best} false-alarm rate ({false_alarms:.0%})", annotation_position="bottom right")
    style(dots, "Share of fault rows caught, by fault (click the legend to add methods)",
          "Fault rows caught", "Fault", percent=True, height=370)
    dots.update_xaxes(dtick=1)
    dots.update_yaxes(range=[0, 1.06])
    dots.update_layout(hovermode="closest")

    top = importance.head(12).iloc[::-1]
    labels = top["tag_name"] + " (" + top["sensor"] + ")"
    importance_chart = go.Figure()
    importance_chart.add_bar(y=labels, x=top["level"], name="Reading level", orientation="h", marker_color=BLUE)
    importance_chart.add_bar(y=labels, x=top["step_change"], name="3-minute change", orientation="h", marker_color=TEAL)
    style(importance_chart, "Sensors the random forest relies on most", x_title="Share of model importance", height=400)
    importance_chart.update_layout(barmode="stack", hovermode="y unified", margin=dict(l=210, r=16, t=52, b=36))
    importance_chart.update_xaxes(tickformat=".0%")

    reliability = go.Figure()
    reliability.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfectly calibrated",
                            line=dict(color=GRAY, dash="dot", width=1.5))
    for (model, group), color in zip(calibration.groupby("model", sort=False), [ORANGE, BLUE, TEAL]):
        reliability.add_scatter(x=group["mean_predicted_prob"], y=group["observed_fault_rate"], mode="lines+markers",
                                name=model, line=dict(color=color, width=2.2), marker=dict(size=8))
    style(reliability, "Does a 70% fault probability mean 70%?", "Share that were faults",
          "Predicted fault probability", percent=True, height=400)
    reliability.update_xaxes(tickformat=".0%", range=[0, 1.02])
    reliability.update_yaxes(range=[0, 1.05])
    reliability.update_layout(hovermode="closest")

    return {
        "dots": dots,
        "importance": importance_chart,
        "reliability": reliability,
        "table": rows,
        "scores": scores,
        "best": best,
        "rule": rule,
        "blind_spots": blind_spots,
        "top_tags": importance["tag_name"].head(3).tolist(),
        "rows": len(predictions),
    }


def main():
    market = export_results()
    forecast = forecast_results()
    faults = fault_results()

    miss, best_forecast = forecast["miss"], forecast["best"]
    baseline_miss = miss["same_as_last_month"]
    best_forecast_name = FORECAST_METHODS[best_forecast][0]
    simple_miss = miss["forecast"]
    best_fault, rule = faults["best"], faults["rule"]
    best_score, rule_score = faults["scores"][best_fault], faults["scores"][rule]
    blind = ", ".join(str(fault) for fault in faults["blind_spots"]) or "none"
    top_tags = ", ".join(tag.lower() for tag in faults["top_tags"])

    tiles = [
        (f"{market['share']:.0%}", f"of U.S. LNG exports came from Venture Global's two plants in the last 12 months"),
        (f"{market['summer']:+.0%}", "average June–September exports against trend: the summer dip a plan must expect"),
        (f"{miss[best_forecast] / baseline_miss - 1:+.0%}",
         f"forecast miss against “same as last month” ({miss[best_forecast]:.1f} vs {baseline_miss:.1f} Bcf)"),
        (f"{best_score['caught']:.0%} / {best_score['cleared']:.0%}",
         f"fault rows caught / normal rows cleared by the best detector ({rule_score['caught']:.0%} / {rule_score['cleared']:.0%} for the rule)"),
    ]
    tile_html = "".join(f'<div class="tile"><b>{value}</b><span>{label}</span></div>' for value, label in tiles)

    questions = pd.DataFrame({
        "Question": [
            "<b>1. How much LNG do Venture Global's plants ship, and how fast did that grow?</b>",
            "<b>2. What regular pattern should a monthly production plan expect?</b>",
            "<b>3. Can next month's exports be forecast better than “same as last month”?</b>",
            "<b>4. Can sensor data flag a process fault without flooding the control room with false alarms?</b>",
            "<b>5. When an alert fires, which sensors should an engineer check first?</b>",
        ],
        "Why it matters": [
            "Sets the stakes: a small gain in train output is a large volume nationally",
            "A plan that ignores the season over-promises every summer",
            "Daily targets and cargo plans start from a volume forecast",
            "Unplanned downtime and safety; too many false alarms and operators stop trusting alerts",
            "An alert is only useful if it points to where to look",
        ],
        "Data": ["EIA exports by terminal", "EIA exports, NOAA Lake Charles temperature", "EIA exports, NOAA temperature",
                 "Tennessee Eastman sensor runs", "Tennessee Eastman sensor runs"],
        "Answer": [
            f"About {market['share']:.0%} of U.S. exports (~{market['vg_bcf']:.0f} Bcf a month), from zero in early 2022",
            f"Exports run {market['summer']:+.0%} against trend in June–September and {market['winter']:+.0%} in the coldest months",
            f"Yes. {best_forecast_name} misses by {miss[best_forecast]:.1f} Bcf a month against {baseline_miss:.1f} for the baseline",
            f"Partly. The best detector, {best_fault}, catches {best_score['caught']:.0%} of fault rows and clears {best_score['cleared']:.0%} of normal rows. Faults {blind} stay hard to see",
            f"The model leans most on the {top_tags}",
        ],
        "See": ['<a href="#market">Results A</a>', '<a href="#market">Results A</a>', '<a href="#forecast">Results B</a>',
                '<a href="#faults">Results C</a>', '<a href="#faults">Results C</a>'],
    })

    sources = pd.DataFrame({
        "Source": [
            '<a href="https://github.com/camaramm/tennessee-eastman-profBraatz">Tennessee Eastman process benchmark</a> (Braatz, University of Illinois)',
            '<a href="https://www.eia.gov/dnav/ng/hist/n9133us2m.htm">EIA U.S. LNG exports</a>, monthly',
            '<a href="https://www.eia.gov/opendata/bulk/">EIA LNG exports by terminal</a>, monthly',
            '<a href="https://www.eia.gov/dnav/ng/hist/rngwhhdm.htm">EIA Henry Hub spot price</a>, monthly',
            '<a href="https://www.ncei.noaa.gov/">NOAA monthly climate summary</a>, Lake Charles airport',
            '<a href="https://www.ncei.noaa.gov/stormevents/">NOAA Storm Events</a>, Louisiana',
            '<a href="https://github.com/microsoft/sqlworkshops">Microsoft predictive-maintenance sample</a>',
        ],
        "Stands in for": [
            "Plant historian tags (OSIsoft PI): 52 flows, pressures, temperatures and valves every 3 minutes, with labelled faults",
            "Plant production (LNG rundown)",
            "Terminal throughput and the cargo schedule",
            "Feed-gas price and supply",
            "Ambient air temperature at the trains",
            "Weather risk around the Gulf terminals",
            "Work orders and equipment failures",
        ],
        "Used for": ["Fault detection (Results C)", "Forecast (Results B)", "Market context (Results A)", "EDA only",
                     "Seasonal pattern and forecast inputs", "EDA only", "EDA only"],
    })

    workflow = [
        ("1. Measure", "Sensors write flows, pressures and temperatures to the OSIsoft PI historian",
         "<code>download.py</code> lands the public files in <code>data/raw</code>"),
        ("2. Ingest", "Azure Data Factory copies PI tags and schedules into Azure Data Lake (raw layer)",
         "Raw files kept as downloaded"),
        ("3. Clean &amp; build features", "Databricks (PySpark, SQL) cleans, aligns and joins the data into curated tables",
         "<code>prepare.py</code>: PySpark sets the normal band and flags alerts; DuckDB SQL joins the monthly tables"),
        ("4. Train &amp; validate", "Forecasting and classification models, tested against simple baselines",
         "<code>model.py</code>: rule vs random forest vs neural network; forecasts tested month by month"),
        ("5. Score", "Batch scoring for daily and monthly plans; streaming scoring for live alerts",
         "Batch: a GitHub Action reruns everything on the 15th of each month"),
        ("6. Decide", "Production engineers combine alerts, forecasts and the schedule to set each train's target",
         "This dashboard: what to inspect, and what volume to plan for"),
    ]
    workflow_html = "".join(
        f'<div class="step"><b>{name}</b><p>{at_plant}</p><p class="here">This project: {here}</p></div>'
        for name, at_plant, here in workflow
    )

    approach = [
        ("Understand the problem", "#problem"),
        ("Frame the questions", "#questions"),
        ("Find the data", "#data"),
        ("Build the workflow", "#workflow"),
        ("Present the results", "#market"),
    ]
    approach_html = "".join(f'<a class="chip" href="{href}"><span>{i}</span>{name}</a>'
                            for i, (name, href) in enumerate(approach, start=1))

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LNG Sensor to Decision</title>
<style>
  :root {{ --ink: #0b0b0b; --muted: #4e4d49; --line: #e4e3de; --surface: #f8f8f6; --card: #ffffff;
          --blue: #2a78d6; --navy: #163e73; --teal: #159a80; --soft-blue: #eef5fc; --soft-teal: #eaf7f4; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--surface); color: var(--ink); font: 15px/1.6 system-ui, -apple-system, sans-serif; }}
  main {{ max-width: 1100px; margin: 0 auto; padding: 32px 16px 72px; }}
  h1 {{ margin: 0 0 8px; font-size: 30px; line-height: 1.2; letter-spacing: -0.02em; overflow-wrap: anywhere; }}
  h2 {{ margin: 48px 0 8px; font-size: 22px; border-bottom: 2px solid var(--line); padding-bottom: 8px; scroll-margin-top: 16px; }}
  h3 {{ margin: 24px 0 6px; font-size: 17px; color: var(--navy); }}
  p {{ color: var(--muted); max-width: 820px; margin: 0 0 14px; }}
  a {{ color: var(--blue); }}
  .subtitle {{ font-size: 16px; max-width: 880px; }}
  .chips {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 18px 0 8px; }}
  .chip {{ display: inline-flex; align-items: center; gap: 8px; padding: 6px 12px; border: 1px solid var(--line); border-radius: 999px;
           background: var(--card); color: var(--ink); text-decoration: none; font-size: 14px; }}
  .chip span {{ display: inline-grid; place-items: center; width: 20px; height: 20px; border-radius: 50%; background: var(--navy); color: #fff; font-size: 12px; }}
  .tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; margin: 24px 0 8px; }}
  .tile {{ background: var(--card); border: 1px solid var(--line); border-top: 3px solid var(--blue); border-radius: 10px; padding: 16px; }}
  .tile b {{ display: block; font-size: 28px; margin-bottom: 4px; }}
  .tile span {{ color: var(--muted); font-size: 13px; line-height: 1.4; display: block; }}
  .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 16px; margin: 14px 0; overflow-x: auto; }}
  .card .plotly-graph-div {{ width: 100% !important; }}
  .grid-2 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(460px, 100%), 1fr)); gap: 14px; }}
  .grid-3 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(300px, 100%), 1fr)); gap: 14px; }}
  .box {{ background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; }}
  .box b {{ display: block; color: var(--navy); margin-bottom: 4px; }}
  .box p {{ margin: 0; font-size: 14px; }}
  .steps {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(240px, 100%), 1fr)); gap: 12px; margin: 14px 0; }}
  .step {{ background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; }}
  .step b {{ display: block; color: var(--navy); margin-bottom: 4px; }}
  .step p {{ font-size: 14px; margin: 0 0 8px; }}
  .step .here {{ color: var(--teal); margin: 0; font-size: 13px; }}
  .note {{ font-size: 13px; color: var(--muted); }}
  .callout {{ background: var(--soft-blue); border-left: 4px solid var(--blue); border-radius: 6px; padding: 12px 16px; margin: 12px 0 16px; font-size: 14px; }}
  .callout.action {{ background: var(--soft-teal); border-left-color: var(--teal); }}
  .callout p, .callout li {{ color: #1e293b; margin: 0 0 6px; }}
  .callout ul {{ margin: 6px 0 0; padding-left: 20px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13.5px; }}
  th {{ background: #f3f2ee; font-weight: 600; text-align: left; padding: 10px 12px; border-bottom: 2px solid var(--line); }}
  td {{ text-align: left; vertical-align: top; padding: 9px 12px; border-bottom: 1px solid var(--line); color: #2e2d2a; }}
  tr:last-child td {{ border-bottom: none; }}
  footer {{ margin-top: 56px; padding-top: 20px; border-top: 1px solid var(--line); color: var(--muted); font-size: 13px; }}
</style>
</head>
<body>
<main>
<h1>From Plant Sensors to LNG Production Decisions</h1>
<p class="subtitle">
  This project rebuilds, on public data, the path an LNG operator follows from raw plant readings to a production decision.
  Sensor readings are cleaned and checked against a normal operating band, models flag process faults and forecast next month's
  exports, and the results are scored against simple baselines so an engineer can see what is worth trusting.
  The whole pipeline reruns on the 15th of each month. Latest export data: <b>{market['latest']:%B %Y}</b>.
</p>
<div class="chips">{approach_html}</div>
<div class="tiles">{tile_html}</div>

<h2 id="problem">1. The problem</h2>
<p>
  An LNG plant earns money by keeping its liquefaction trains producing as much as they safely can. Calcasieu Pass runs 18 mid-scale trains,
  and Plaquemines is adding more. At about {market['vg_bcf']:.0f} Bcf a month, a 1% gain in output is roughly {market['vg_bcf'] / 100:.1f} Bcf a month.
  Three issues decide that output:
</p>
<div class="grid-3">
  <div class="box"><b>Production efficiency</b><p>Get more LNG from the trains already running, and know what each month should deliver.</p></div>
  <div class="box"><b>Downtime and safety</b><p>Catch a process fault early, while keeping false alarms low enough that operators trust the alerts.</p></div>
  <div class="box"><b>Planning</b><p>Daily targets, maintenance and cargo schedules all depend on a volume forecast that matches what the plant delivers.</p></div>
</div>

<h2 id="questions">2. Questions this dashboard answers</h2>
<p>Each question comes from one of the issues above. The answers update every month from the latest data.</p>
<div class="card">{table(questions)}</div>

<h2 id="data">3. Data sources</h2>
<p>Real plant telemetry is private, so each public source stands in for one internal feed.</p>
<div class="card">{table(sources)}</div>

<h2 id="workflow">4. Workflow and tools</h2>
<h3>How the machine-learning workflow typically runs at an LNG operator</h3>
<p class="note">
  This is an assumed workflow, pieced together from public job postings (OSIsoft PI, Azure Data Factory, Azure Data Lake, Databricks,
  forecasting and classification). It is not an internal diagram. Each step shows the matching part of this project.
</p>
<div class="steps">{workflow_html}</div>
<div class="callout">
  <p><b>Tools used:</b> Python, PySpark (standing in for Databricks), DuckDB SQL, pandas, scikit-learn (random forest, neural network),
  Plotly, and GitHub Actions for the monthly rebuild. Tables follow raw → silver (cleaned) → gold (joined, model outputs).</p>
  <p><b>Rule for every model:</b> it is scored only on data it never trained on, and it has to beat a simple baseline that an engineer would use anyway.</p>
</div>

<h2 id="market">5. Results A: what drives U.S. LNG volume</h2>
<p>
  Venture Global's plants went from nothing in early 2022 to about <b>{market['share']:.0%} of U.S. LNG exports</b>.
  Exports also follow the season: they run <b>{market['summer']:+.0%}</b> against trend from June to September and
  <b>{market['winter']:+.0%}</b> in the coldest months. That is consistent with liquefaction losing capacity in hot air, as compressors
  and air coolers work harder, and with maintenance being scheduled outside winter.
</p>
<div class="card">{chart(market['stack'], first=True)}</div>
<div class="card">{chart(market['season'])}</div>

<h2 id="forecast">6. Results B: next-month export forecast</h2>
<p>
  <b>What is compared:</b> six ways to forecast next month's U.S. LNG exports. <b>What counts as actual:</b> EIA's reported exports for that month.
  <b>How:</b> each of the last {forecast['months']} months ({forecast['first']:%b %Y} – {forecast['last']:%b %Y}) is forecast using only the months before it,
  the way a real plan is made.
</p>
<div class="card">{chart(forecast['lines'])}</div>
<div class="card">{table(forecast['table'])}</div>
<div class="callout">
  <ul>
    <li><b>Counting days does most of the work.</b> Scaling last month's daily rate by the days in the coming month cuts the miss from
      {baseline_miss:.1f} to {miss['same_daily_rate']:.1f} Bcf.</li>
    <li><b>The season adds a little.</b> Blending in the usual calendar-month change brings it to {simple_miss:.1f} Bcf ({miss['forecast'] / baseline_miss - 1:+.0%}).</li>
    <li><b>The learned models are close to the simple blend.</b> The best method, {best_forecast_name}, misses by {miss[best_forecast]:.1f} Bcf,
      {abs(simple_miss - miss[best_forecast]):.1f} Bcf from the blend. With {forecast['months']} test months, a gap that small is within noise,
      so the simple blend is the easier one to explain and maintain.</li>
  </ul>
</div>
<div class="callout action">
  <p><b>What a planner does with it:</b> plan around the forecast, not the baseline, and expect the summer dip. The remaining miss comes from new trains
  ramping up and cargo timing, which national totals cannot see. Inside a plant, adding commissioning dates and maintenance windows is the next step.</p>
</div>

<h2 id="faults">7. Results C: sensor fault detection</h2>
<p>
  <b>What is compared:</b> a 3-sigma alarm rule against three machine-learning models. <b>What counts as actual:</b> the benchmark's labels, where a test run
  is normal for 8 hours and then a known fault is switched on. <b>How:</b> models train on one set of runs and are scored on {faults['rows']:,} rows of
  separate test runs they never saw.
</p>
<div class="card">{table(faults['table'])}</div>
<div class="card">{chart(faults['dots'])}</div>
<div class="grid-2">
  <div class="card">{chart(faults['importance'])}</div>
  <div class="card">{chart(faults['reliability'])}</div>
</div>
<div class="callout">
  <ul>
    <li><b>The rule is a strong baseline.</b> It catches {rule_score['caught']:.0%} of fault rows and clears {rule_score['cleared']:.0%} of normal rows.</li>
    <li><b>The best detector is the {best_fault}:</b> {best_score['caught']:.0%} caught and {best_score['cleared']:.0%} cleared. Averaging over 15 minutes
      matters as much as the model, because a real fault persists and noise does not.</li>
    <li><b>Some faults stay hidden.</b> Faults {blind} are caught less than 60% of the time by every method. The benchmark literature reports the same:
      they barely move the 52 sensors. Better instruments or more normal history would help more than a bigger model.</li>
    <li><b>Where to look first:</b> the {top_tags} carry the most weight in the random forest.</li>
  </ul>
</div>
<div class="callout action">
  <p><b>What an engineer does with it:</b> treat an alert as a reason to inspect the train before raising its target, start with the sensors above,
  and only page the field crew when the alert holds for several readings.</p>
</div>

<h2>8. Limits</h2>
<p>
  Public data stands in for plant data, so the numbers show the method, not Venture Global's actual performance. The fault benchmark is a chemical process,
  not an LNG train, and "normal" is learned from a single 25-hour run. The forecast uses national totals, not train-level rundown. Each limit has a direct
  fix inside a plant: PI tags with maintenance labels, longer normal history, and train-level volumes.
</p>

<footer>
  Pipeline: <code>download.py</code> → <code>prepare.py</code> (PySpark, DuckDB) → <code>model.py</code> (scikit-learn) → <code>dashboard.py</code>, rebuilt monthly by GitHub Actions.
  Source code: <a href="https://github.com/mhadeli/lng-sensor-to-decision">github.com/mhadeli/lng-sensor-to-decision</a>.
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
