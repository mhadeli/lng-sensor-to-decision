"""Build a plain-language portfolio dashboard from evaluated model outputs."""
import json
from html import escape
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

ROOT = Path(__file__).resolve().parent
GOLD = ROOT / "data" / "gold"
SITE = ROOT / "site"
NAMES = {"baseline": "Simple baseline", "mlp": "Neural network (MLP)", "xgboost": "XGBoost"}
COLORS = {"baseline": "#949c9c", "mlp": "#296abd", "xgboost": "#087f73"}


def chart(fig, chart_id, ytitle):
    fig.update_layout(template="plotly_white", height=340, margin=dict(l=45, r=20, t=20, b=50),
                      font=dict(family="Arial, sans-serif", color="#405051", size=12),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=-0.22), hovermode="x unified")
    fig.update_yaxes(title=ytitle, gridcolor="#e6eae6", zeroline=False)
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=chart_id,
                       config={"displayModeBar": False, "responsive": True})


def table(rows):
    return '<div class="table-wrap">' + pd.DataFrame(rows).to_html(index=False, border=0, escape=True) + '</div>'


def main():
    evaluation = json.loads((GOLD / "evaluation.json").read_text())
    faults, exports = evaluation["faults"], evaluation["exports"]
    chosen_fault, chosen_export = faults["selected"], exports["selected"]
    f, b = faults["test"][chosen_fault], faults["test"]["baseline"]
    e, eb = exports["test"][chosen_export], exports["test"]["baseline"]
    predictions = pd.read_csv(GOLD / "fault_predictions.csv")
    forecast = pd.read_csv(GOLD / "volume_forecast.csv")
    importance = pd.read_csv(GOLD / "feature_importance.csv").head(8).iloc[::-1]
    caught_delta = (f["caught"] - b["caught"]) * 100
    alarm_delta = (f["false_alarm"] - b["false_alarm"]) * 100
    improvement = 1 - e["mae_bcf"] / eb["mae_bcf"]
    fault_message = (f"Compared with the rule, this approach catches {abs(caught_delta):.1f} percentage points "
                     f"{'more' if caught_delta >= 0 else 'fewer'} abnormal readings, with "
                     f"{abs(alarm_delta):.1f} points {'more' if alarm_delta >= 0 else 'fewer'} false alarms.")
    if chosen_fault == "baseline":
        fault_message = "The simple rule won on validation. The learned models remain visible below for an honest comparison."
    export_message = (f"The selected approach has {abs(improvement):.0%} {'less' if improvement >= 0 else 'more'} error than "
                      "carrying the previous month’s daily export rate forward.")
    if chosen_export == "baseline":
        export_message = "The daily-rate baseline won on validation. Added model complexity did not earn a place in the headline forecast."

    bars = go.Figure()
    for label, metric, color in [("Abnormal readings detected", "caught", "#087f73"),
                                  ("Normal readings incorrectly flagged", "false_alarm", "#c27c44")]:
        bars.add_bar(x=[NAMES[k] for k in NAMES], y=[faults["test"][k][metric] for k in NAMES],
                     name=label, marker_color=color, text=[f'{faults["test"][k][metric]:.1%}' for k in NAMES],
                     textposition="outside", hovertemplate="%{x}: %{y:.1%}<extra>" + label + "</extra>")
    bars.update_layout(barmode="group")
    bars.update_yaxes(tickformat=".0%", range=[0, 1.08])
    lines = go.Figure()
    lines.add_scatter(x=forecast.month, y=forecast.actual / 1000, name="Actual exports",
                      line=dict(color="#203638", width=3))
    for kind in NAMES:
        lines.add_scatter(x=forecast.month, y=forecast[kind] / 1000, name=NAMES[kind],
                          visible=True if kind in {"baseline", chosen_export} else "legendonly",
                          line=dict(color=COLORS[kind], width=2, dash="dot" if kind == "baseline" else "solid"))
    fault_rows = [{"Approach": NAMES[k] + (" · selected" if k == chosen_fault else ""),
                   "Abnormal detected ↑": f'{faults["test"][k]["caught"]:.1%}',
                   "False alarms ↓": f'{faults["test"][k]["false_alarm"]:.1%}',
                   "Balanced accuracy ↑": f'{faults["test"][k]["balanced_accuracy"]:.1%}',
                   "ROC-AUC ↑": f'{faults["test"][k]["auc"]:.3f}' if faults["test"][k]["auc"] is not None else "—"}
                  for k in NAMES]
    export_rows = [{"Approach": NAMES[k] + (" · selected" if k == chosen_export else ""),
                    "Average error ↓": f'{exports["test"][k]["mae_bcf"]:.1f} Bcf',
                    "Average percentage error ↓": f'{exports["test"][k]["mape"]:.1%}'} for k in NAMES]
    per_fault = predictions[predictions.is_fault == 1].groupby("fault_id")[f"{chosen_fault}_predicted"].mean()
    difficult = ", ".join(str(i) for i in per_fault[per_fault < .5].index) or "None below 50% detection"
    sensor_chart = go.Figure(go.Bar(x=importance.importance, y=importance.tag_name,
                                   orientation="h", marker_color="#087f73"))
    sensor_chart.update_layout(margin=dict(l=220))
    sensor_chart.update_xaxes(tickformat=".0%")
    # Keep the chart label area readable on smaller screens through the scroll container.
    sensor_html = chart(sensor_chart, "sensor-importance", "")
    per_fault_rows = [{"Fault scenario": int(i), "Abnormal readings detected": f"{rate:.1%}"}
                      for i, rate in per_fault.items()]
    html = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LNG · From data to decisions</title><script src="plotly.min.js"></script>
<style>
:root{{--ink:#203638;--muted:#5e6f70;--teal:#087f73;--line:#dce3de;--paper:#f5f6f1}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.65 system-ui,-apple-system,sans-serif}}
main{{max-width:1120px;margin:auto;padding:36px 28px 60px}}nav{{display:flex;justify-content:space-between;gap:20px;align-items:center;padding-bottom:30px;font-size:13px}}nav span{{letter-spacing:.13em;font-weight:700}}nav a{{margin-left:22px;color:var(--muted);text-decoration:none}}
header{{padding:32px 0 34px;max-width:830px}}.eyebrow{{color:var(--teal);text-transform:uppercase;letter-spacing:.14em;font-size:12px;font-weight:700}}h1{{font-size:clamp(36px,5vw,58px);line-height:1.1;font-weight:650;letter-spacing:-.045em;margin:18px 0 22px}}header p{{font-size:19px;color:var(--muted);max-width:740px}}h2{{font-size:26px;line-height:1.25;letter-spacing:-.025em;margin:8px 0 12px}}h3{{font-size:16px;margin:0 0 8px}}p{{margin:10px 0 18px}}.muted,.note{{color:var(--muted)}}.note{{font-size:13px}}.steps{{display:grid;grid-template-columns:repeat(3,1fr);gap:28px;border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:24px 0;margin:0 0 36px}}.steps p{{font-size:14px;color:var(--muted);margin:0}}.steps b{{color:var(--teal);margin-right:8px}}section{{scroll-margin-top:20px;background:white;border:1px solid var(--line);border-radius:16px;padding:30px;margin-bottom:24px}}.section-head{{max-width:800px}}.section-head p{{color:var(--muted)}}.metrics{{display:grid;grid-template-columns:repeat(3,1fr);gap:24px;padding:24px 0;border-top:1px solid var(--line)}}.metric strong{{display:block;font-size:36px;letter-spacing:-.04em;line-height:1.25}}.metric span{{font-size:13px;color:var(--muted)}}.takeaway{{background:#eff6f2;border-left:3px solid var(--teal);padding:14px 18px;font-size:15px;margin:16px 0 24px}}.chart{{min-width:0}}details{{border-top:1px solid var(--line);padding:18px 0 0;margin-top:16px}}summary{{cursor:pointer;font-size:14px;font-weight:650;color:var(--teal)}}details p{{font-size:14px}}.table-wrap{{overflow-x:auto;margin:20px 0}}table{{width:100%;border-collapse:collapse;font-size:13px;text-align:left}}th,td{{padding:13px 12px;border-bottom:1px solid var(--line);text-align:left!important}}th{{color:var(--muted);font-weight:600;background:#f7f9f6}}.transfer{{display:grid;grid-template-columns:repeat(3,1fr);gap:28px;margin-top:25px}}.transfer p{{font-size:14px;color:var(--muted)}}a{{color:var(--teal)}}footer{{font-size:12px;color:var(--muted);padding:10px 2px}}a:focus-visible,summary:focus-visible{{outline:3px solid #296abd;outline-offset:4px}}
@media(max-width:680px){{main{{padding:20px 14px}}nav a{{margin-left:10px}}nav span{{font-size:10px}}section{{padding:22px 16px}}.steps,.transfer{{grid-template-columns:1fr;gap:18px}}.metrics{{gap:12px}}.metric strong{{font-size:27px}}.metric span{{font-size:12px}}h2{{font-size:23px}}header p{{font-size:17px}}.sensor-chart{{overflow-x:auto}}.sensor-chart>div{{min-width:640px}}}}
</style></head><body><main>
<nav aria-label="Page sections"><span>DATA SCIENCE PORTFOLIO</span><div><a href="#detect">Detection</a><a href="#plan">Planning</a><a href="#apply">Application</a></div></nav>
<header><div class="eyebrow">LNG · From data to decisions</div><h1>Spot process changes.<br>Make better plans.</h1><p>I built a reproducible data science workflow to answer two operational questions: can we identify abnormal sensor behavior, and how accurately can we forecast next month’s LNG exports?</p><p class="note">A demonstration using simulated process readings and public U.S. export data. Results measure performance on these datasets, not an operating LNG facility.</p></header>
<div class="steps"><div><h3><b>01</b> Prepare the data</h3><p>Validate sensor histories and monthly exports. Keep original data separate from model-ready tables.</p></div><div><h3><b>02</b> Compare three approaches</h3><p>A simple baseline, a small neural network, and XGBoost. Choose settings before evaluating the final test data.</p></div><div><h3><b>03</b> Explain the trade-offs</h3><p>Show detection, false alarms, and forecast error in terms an operations team can use.</p></div></div>
<section id="detect"><div class="section-head"><div class="eyebrow">01 / Process monitoring</div><h2>Which readings deserve a closer look?</h2><p>Identify abnormal behavior across 52 simulated process sensors. The practical trade-off: catch more problems without overwhelming operators with false alarms.</p></div>
<div class="metrics"><div class="metric"><strong>{f['caught']:.0%}</strong><span>abnormal readings detected</span></div><div class="metric"><strong>{f['false_alarm']:.1%}</strong><span>normal readings incorrectly flagged</span></div><div class="metric"><strong>{faults['test_rows']:,}</strong><span>readings in separate test runs</span></div></div>
<p class="note">Headline approach: <b>{NAMES[chosen_fault]}</b>, selected on validation data. Rates describe individual readings, not incidents prevented.</p>
<div class="chart" role="img" aria-label="Detection and false alarm rates for the three approaches">{chart(bars, 'fault-comparison', 'Share of readings')}</div>
<div class="takeaway">{fault_message} Use these flags to prioritize investigation; they do not identify the cause of a fault.</div>
<details><summary>Explore the model comparison and evaluation</summary>{table(fault_rows)}<p>The baseline flags any sensor outside its normal training mean ± three standard deviations. Both learned models use sensor levels, changes, and trailing five-reading averages, scaled against normal training operation with absolute deviation features; their scores are also averaged over five readings. This reduces noise but can delay detection.</p><p>Settings and alert thresholds maximize balanced accuracy on the last quarter of each training run, with a five-reading gap. Models are then refitted on all training runs and evaluated on separate test runs. The same fault scenarios appear in training and test; this does not establish performance on new fault types. There is only one normal training run, so validation remains limited. The benchmark test data has been reviewed during development; a fresh prospective evaluation is still needed.</p><p>Scores are model outputs, not calibrated failure probabilities. Balanced accuracy gives detection and correctly clearing normal readings equal weight. ROC-AUC measures ranking across thresholds.</p><p><b>Where performance is weak:</b> fault scenarios with fewer than half of abnormal readings detected: {escape(difficult)}.</p>{table(per_fault_rows)}</details>
<details><summary>See which sensors XGBoost uses most</summary><div class="sensor-chart">{sensor_html}</div><p>Importance is the model’s relative reliance on each sensor, combined across levels, changes, and rolling means. Correlated inputs can share importance. This is not a root-cause diagnosis.</p></details></section>
<section id="plan"><div class="section-head"><div class="eyebrow">02 / Monthly planning</div><h2>How close can next month’s forecast get?</h2><p>Forecast U.S. LNG export volume one month ahead. Each estimate uses only observations from earlier months, then gets compared with the actual outcome.</p></div>
<div class="metrics"><div class="metric"><strong>{e['mae_bcf']:.1f} Bcf</strong><span>average monthly forecast error</span></div><div class="metric"><strong>{e['mape']:.1%}</strong><span>average percentage error</span></div><div class="metric"><strong>{exports['test_months']}</strong><span>months evaluated one at a time</span></div></div>
<p class="note">Headline approach: <b>{NAMES[chosen_export]}</b>, selected on validation data. Test period: {exports['test_first']} to {exports['test_last']}. Bcf = billion cubic feet of gas.</p>
<div class="chart" role="img" aria-label="Actual monthly exports and historical one-month-ahead forecasts">{chart(lines, 'export-forecast', 'Monthly exports (Bcf)')}</div><p class="note">Click a legend label to show or hide an approach. These are historical test forecasts, not a live forecast.</p>
<div class="takeaway">{export_message} This gives a national planning benchmark; a plant-level plan would also need availability, feed gas, and cargo schedules.</div>
<details><summary>Explore forecast accuracy and the backtest</summary>{table(export_rows)}<p>The baseline carries the previous month’s daily export rate into the next month, adjusting for its number of days. MLP and XGBoost learn the change in daily rate using seasonality and lagged growth. Every seasonal feature is calculated using earlier observations only.</p><p>Model settings are selected on {exports['validation_first']} to {exports['validation_last']}, before the final {exports['test_months']}-month evaluation. The selected approach stays fixed; models refit as each month becomes available. Published data can be revised and arrive with a delay. This backtest assumes the prior month is available and does not recreate historical publication vintages.</p><p>Average error is mean absolute error. Percentage error averages each month’s absolute error divided by actual exports. No claim of savings or production gains is made from this retrospective test.</p></details></section>
<section id="apply"><div class="eyebrow">From demonstration to plant use</div><h2>How I would apply the same workflow</h2><p class="muted">Start with a decision and its owner, then agree on the data and evidence needed to support it.</p><div class="transfer"><div><h3>Work with operations</h3><p>Define an actionable fault event, forecast horizon, and acceptable false-alarm rate. Review missed events with engineers and agree on what should trigger a response.</p></div><div><h3>Connect plant context</h3><p>Combine historian readings with train status and maintenance records. For forecasts, add feed-gas constraints, planned outages, and cargo schedules.</p></div><div><h3>Validate before rollout</h3><p>Test on later periods and separate trains. Run alongside existing processes, measure detection delay and alert burden, then monitor drift and retraining needs.</p></div></div>
<details><summary>Review the workflow, sources, and reproducibility</summary><p><b>Implemented:</b> Python ingestion → pandas validation and DuckDB SQL → baseline / MLP / XGBoost → saved predictions and evaluation → this static dashboard. Optional PySpark preparation demonstrates a path to larger data volumes. GitHub Actions rebuilds and publishes after checks pass.</p><p><b>Possible plant deployment:</b> Data Factory ingestion into Azure Data Lake; Databricks for preparation and scheduled scoring; governed outputs for engineering review. This is a proposed adaptation, not a deployed integration.</p><p><b>Sources:</b> <a href="https://github.com/camaramm/tennessee-eastman-profBraatz">Tennessee Eastman process simulation</a> and <a href="https://www.eia.gov/dnav/ng/hist_xls/N9133US2m.xls">EIA monthly U.S. LNG exports</a>. The two datasets answer separate questions and are not joined as if they represented the same facility.</p><p>Rebuild locally with <code>python download.py</code>, <code>python prepare.py</code>, <code>python model.py</code>, and <code>python dashboard.py</code>. Read the project README for evaluation design and limitations.</p></details></section>
<footer>Public-data portfolio · Python / SQL / scikit-learn / XGBoost · All displayed metrics are calculated from saved test predictions.</footer>
</main><script>document.querySelectorAll('details').forEach(d=>d.addEventListener('toggle',()=>{{if(d.open)d.querySelectorAll('.plotly-graph-div').forEach(g=>Plotly.Plots.resize(g));}}));</script></body></html>'''
    SITE.mkdir(exist_ok=True)
    (SITE / "plotly.min.js").write_text(get_plotlyjs())
    (SITE / "index.html").write_text(html)
    print(f"Built {SITE / 'index.html'}")


if __name__ == "__main__":
    main()
