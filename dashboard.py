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
    operations = json.loads((GOLD / "fault_operations.json").read_text())
    operational = operations["models"]
    chosen_fault, chosen_export = faults["selected"], exports["selected"]
    event_score = operational[chosen_fault]
    delay_text = f"{event_score['median_delay_minutes_detected_only']:.0f} min" if event_score["median_delay_minutes_detected_only"] is not None else "—"
    burden_text = f"{event_score['false_alerts_per_24_normal_hours']:.2f}" if event_score["false_alerts_per_24_normal_hours"] is not None else "—"
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

    operational_rows = [{"Approach": NAMES[k],
                         "Events flagged within 30 min": f"{v['events_within_30_min']} / {v['events']}",
                         "Median delay¹": f"{v['median_delay_minutes_detected_only']:.0f} min" if v['median_delay_minutes_detected_only'] is not None else "—",
                         "False alerts / 24 normal hours": f"{v['false_alerts_per_24_normal_hours']:.2f}" if v['false_alerts_per_24_normal_hours'] is not None else "—",
                         "Never flagged": str(v['missed_events'])}
                        for k, v in operational.items()]
    event_details = pd.read_csv(GOLD / "fault_events.csv")
    event_details = event_details[event_details.model == chosen_fault]
    event_rows = [{"Fault scenario": int(row.fault_id),
                   "Time to confirmed alert": f"{row.delay_minutes:.0f} min" if pd.notna(row.delay_minutes) else "Not detected",
                   "Within 30 minutes": "Yes" if row.detected_within_30_min else "No"}
                  for row in event_details.itertuples()]
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
header{{padding:32px 0 34px;max-width:830px}}.eyebrow{{color:var(--teal);text-transform:uppercase;letter-spacing:.14em;font-size:12px;font-weight:700}}h1{{font-size:clamp(36px,5vw,58px);line-height:1.1;font-weight:650;letter-spacing:-.045em;margin:18px 0 22px}}header p{{font-size:19px;color:var(--muted);max-width:740px}}h2{{font-size:26px;line-height:1.25;letter-spacing:-.025em;margin:8px 0 12px}}h3{{font-size:16px;margin:0 0 8px}}p{{margin:10px 0 18px}}.muted,.note{{color:var(--muted)}}.note{{font-size:13px}}.steps{{display:grid;grid-template-columns:repeat(3,1fr);gap:28px;border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:24px 0;margin:0 0 36px}}.steps p{{font-size:14px;color:var(--muted);margin:0}}.steps b{{color:var(--teal);margin-right:8px}}section{{scroll-margin-top:20px;background:white;border:1px solid var(--line);border-radius:16px;padding:30px;margin-bottom:24px}}.section-head{{max-width:800px}}.section-head p{{color:var(--muted)}}.metrics{{display:grid;grid-template-columns:repeat(3,1fr);gap:24px;padding:24px 0;border-top:1px solid var(--line)}}.metric strong{{display:block;font-size:36px;letter-spacing:-.04em;line-height:1.25}}.metric span{{font-size:13px;color:var(--muted)}}.takeaway{{background:#eff6f2;border-left:3px solid var(--teal);padding:14px 18px;font-size:15px;margin:16px 0 24px}}.chart{{min-width:0}}details{{border-top:1px solid var(--line);padding:18px 0 0;margin-top:16px}}li{{margin:10px 0;font-size:14px}}summary{{cursor:pointer;font-size:14px;font-weight:650;color:var(--teal)}}details p{{font-size:14px}}.table-wrap{{overflow-x:auto;margin:20px 0}}table{{width:100%;border-collapse:collapse;font-size:13px;text-align:left}}th,td{{padding:13px 12px;border-bottom:1px solid var(--line);text-align:left!important}}th{{color:var(--muted);font-weight:600;background:#f7f9f6}}.transfer{{display:grid;grid-template-columns:repeat(3,1fr);gap:28px;margin-top:25px}}.transfer p{{font-size:14px;color:var(--muted)}}a{{color:var(--teal)}}footer{{font-size:12px;color:var(--muted);padding:10px 2px}}a:focus-visible,summary:focus-visible{{outline:3px solid #296abd;outline-offset:4px}}
@media(max-width:680px){{main{{padding:20px 14px}}nav{{flex-wrap:wrap}}nav a{{margin-left:0;margin-right:12px}}nav span{{font-size:10px}}section{{padding:22px 16px}}.steps,.transfer{{grid-template-columns:1fr;gap:18px}}.metrics{{gap:12px}}.metric strong{{font-size:27px}}.metric span{{font-size:12px}}h2{{font-size:23px}}header p{{font-size:17px}}.sensor-chart{{overflow-x:auto}}.sensor-chart>div{{min-width:640px}}}}
</style></head><body><main>
<nav aria-label="Page sections"><span>DATA SCIENCE PORTFOLIO</span><div><a href="#workflow">Workflow</a><a href="#detect">Detection</a><a href="#plan">Planning</a><a href="#apply">Application</a></div></nav>
<header><div class="eyebrow">LNG · From data to decisions</div><h1>Spot process changes.<br>Make better plans.</h1><p>I built a reproducible data science workflow to answer two operational questions: can we identify abnormal sensor behavior, and how accurately can we forecast next month’s LNG exports?</p><p class="note">A demonstration using simulated process readings and public U.S. export data. Results measure performance on these datasets, not an operating LNG facility.</p></header>
<section id="workflow"><div class="eyebrow">How I built it</div><h2>From source files to evaluated models</h2>
<div class="steps"><div><h3><b>01</b> Ingest</h3><p>I downloaded Tennessee Eastman sensor files and EIA monthly LNG exports with Python. Raw files are cached separately so each transformation can be reproduced.</p></div><div><h3><b>02</b> Preprocess</h3><p>I validated sensor readings and monthly volumes, assigned fault labels, and sorted time records. Features describe recent changes using only information already available.</p></div><div><h3><b>03</b> Model &amp; evaluate</h3><p>I compared a baseline, MLP, and XGBoost. Validation chooses settings; separate tests measure forecast error, missed readings, detection delay, and alert burden.</p></div></div>
<details><summary>See the data preparation and modeling steps</summary>
<ol><li><b>Acquire and retain:</b> download the sensor ZIP and export workbook into the raw layer. Extract individual runs and convert monthly exports to a consistent CSV; refresh exports on demand or in CI.</li>
<li><b>Validate and label:</b> check for 52 complete sensor columns and duplicate run steps. Label training faults from their start and test faults from step 160. Reject duplicate months, missing or nonpositive volumes; the modeling step also checks that months are consecutive.</li>
<li><b>Build features:</b> use sensor levels, one-step changes, and trailing five-reading means. Fit normal-operation scaling inside each training fit. For exports, adjust for month length and calculate lagged growth and past-only seasonal changes.</li>
<li><b>Train and compare:</b> balance sensor classes and tune a small set of settings on the final quarter of training runs, separated by a gap. Use 12 earlier months to validate forecasts before the 24-month walk-forward test.</li>
<li><b>Evaluate and communicate:</b> save predictions and metrics to the curated layer. Apply the same three-flag event policy to all detectors, document trade-offs, and generate this dashboard. Tests check temporal leakage and metric consistency before CI publishes.</li></ol>
<p class="note">Python / pandas for ingestion and preparation, DuckDB for ordering monthly data with SQL, scikit-learn and XGBoost for modeling. Optional PySpark provides an alternative sensor-preparation path. The default build uses {faults['train_rows'] + faults['test_rows']:,} sensor readings; the two datasets are evaluated separately.</p></details></section>
<section id="detect"><div class="section-head"><div class="eyebrow">01 / Process monitoring</div><h2>Which readings deserve a closer look?</h2><p>Identify abnormal behavior across 52 simulated process sensors. The practical trade-off: catch more problems without overwhelming operators with false alarms.</p></div>
<div class="takeaway"><b>Recommendation: keep existing operational rules during a pilot.</b> The validation winner is a research candidate, not an approved replacement. Operations must agree on acceptable missed events and detection delay, then verify performance on independent plant history.</div>
<div class="metrics"><div class="metric"><strong>{event_score['events_within_30_min']} / {event_score['events']}</strong><span>fault events flagged within 30 minutes</span></div><div class="metric"><strong>{delay_text}</strong><span>median delay for detected events¹</span></div><div class="metric"><strong>{burden_text}</strong><span>false alerts per 24 normal operating hours</span></div></div>
<p class="note">Research candidate: <b>{NAMES[chosen_fault]}</b>, selected on reading-level validation. Event metrics below evaluate a fixed policy afterward; they do not change the selected model or threshold.</p>
{table(operational_rows)}
<p class="note">¹ Median includes detected events only; “never flagged” counts the rest. A confirmed alert requires three consecutive flagged readings. The 30-minute window is an illustrative evaluation target, not an agreed plant requirement. False-alert rates combine {event_score['normal_hours']:.0f} normal simulation hours across runs.</p>
<details><summary>How event detection and alert burden are measured</summary><p>Each continuous fault-labelled interval is one event. A new alert episode begins when flags switch on and confirms at its third consecutive reading. Detection delay runs from fault onset to confirmation, using the three-minute sampling interval. An alert already active before onset does not count as a new detection; a single clear reading resets the episode.</p><p>False alerts are episodes whose first three readings are all normal. Counts are divided by total normal exposure and scaled to 24 hours, not calendar days. No acknowledgement or operator-response data is available. Brief flags are excluded consistently for every approach.</p><p>A fault can eventually be flagged after a long delay or by chance. Eventual detection over a long simulated run is not evidence of timely intervention or downtime avoided. These event results are descriptive; they were not used to retune the model.</p>{table(event_rows)}</details>
<details><summary>Explore reading-level results and model evaluation</summary>
<p class="note">Across {faults['test_rows']:,} test readings, the candidate flags {f['caught']:.1%} of abnormal readings and incorrectly flags {f['false_alarm']:.1%} of normal readings.</p>
<div class="chart" role="img" aria-label="Detection and false alarm rates for the three approaches">{chart(bars, 'fault-comparison', 'Share of readings')}</div>
<p>{fault_message} Use flags to prioritize investigation; they do not identify the cause of a fault.</p>
{table(fault_rows)}<p>The baseline flags any sensor outside its normal training mean ± three standard deviations. Both learned models use sensor levels, changes, and trailing five-reading averages, scaled against normal training operation with absolute deviation features; their scores are also averaged over five readings. This reduces noise but can delay detection.</p><p>Settings and alert thresholds maximize balanced accuracy on the last quarter of each training run, with a five-reading gap. Models are then refitted on all training runs and evaluated on separate test runs. The same fault scenarios appear in training and test; this does not establish performance on new fault types. There is only one normal training run, so validation remains limited. The benchmark test data has been reviewed during development; a fresh prospective evaluation is still needed.</p><p>Scores are model outputs, not calibrated failure probabilities. Balanced accuracy gives detection and correctly clearing normal readings equal weight. ROC-AUC measures ranking across thresholds.</p><p><b>Where performance is weak:</b> fault scenarios with fewer than half of abnormal readings detected: {escape(difficult)}.</p>{table(per_fault_rows)}</details>
<details><summary>See which sensors XGBoost uses most</summary><div class="sensor-chart">{sensor_html}</div><p>Importance is the model’s relative reliance on each sensor, combined across levels, changes, and rolling means. Correlated inputs can share importance. This is not a root-cause diagnosis.</p></details></section>
<section id="plan"><div class="section-head"><div class="eyebrow">02 / Monthly planning</div><h2>How close can next month’s forecast get?</h2><p>This is a national export forecast, not a plant production forecast. It estimates U.S. LNG export volume one month ahead. Each estimate uses only observations from earlier months, then gets compared with the actual outcome.</p></div>
<div class="metrics"><div class="metric"><strong>{e['mae_bcf']:.1f} Bcf</strong><span>average monthly forecast error</span></div><div class="metric"><strong>{e['mape']:.1%}</strong><span>average percentage error</span></div><div class="metric"><strong>{exports['test_months']}</strong><span>months evaluated one at a time</span></div></div>
<p class="note">Headline approach: <b>{NAMES[chosen_export]}</b>, selected on validation data. Test period: {exports['test_first']} to {exports['test_last']}. Bcf = billion cubic feet of gas.</p>
<div class="chart" role="img" aria-label="Actual monthly exports and historical one-month-ahead forecasts">{chart(lines, 'export-forecast', 'Monthly exports (Bcf)')}</div><p class="note">Click a legend label to show or hide an approach. These are historical test forecasts, not a live forecast.</p>
<div class="takeaway">{export_message} This gives a national planning benchmark; a plant-level plan would also need availability, feed gas, and cargo schedules.</div>
<details><summary>How I would adapt this to a plant production forecast</summary><p>Agree with the production planner on a daily LNG rundown target per train, in consistent plant units such as tonnes per day, and a forecast cutoff. National export volume is not a label for train output and cannot be directly converted into a production forecast.</p><p>Join historian production totals to train availability, operating mode, feed-gas supply and quality, ambient conditions, and maintenance schedules using train IDs and timestamps. Include cargo schedules when forecasting shipments or inventory. Use only schedule versions known at the forecast cutoff.</p><p>Rebuild the baseline from the last comparable operating period, then test MLP and XGBoost on the same future periods and held-out trains. Compare error with the existing planning method, including outage and ramp-up periods, before proposing a planning change.</p></details>
<details><summary>Explore forecast accuracy and the backtest</summary>{table(export_rows)}<p>The baseline carries the previous month’s daily export rate into the next month, adjusting for its number of days. MLP and XGBoost learn the change in daily rate using seasonality and lagged growth. Every seasonal feature is calculated using earlier observations only.</p><p>Model settings are selected on {exports['validation_first']} to {exports['validation_last']}, before the final {exports['test_months']}-month evaluation. The selected approach stays fixed; models refit as each month becomes available. Published data can be revised and arrive with a delay. This backtest assumes the prior month is available and does not recreate historical publication vintages.</p><p>Average error is mean absolute error. Percentage error averages each month’s absolute error divided by actual exports. No claim of savings or production gains is made from this retrospective test.</p></details></section>
<section id="apply"><div class="eyebrow">Proposed plant pilot · not yet implemented</div><h2>A small pilot with clear decision gates</h2><p class="muted">Start with one train and a named operations owner. Keep existing alerts and planning processes active while gathering evidence.</p>
<div class="transfer"><div><h3>1. Define the decision</h3><p>Operations and reliability engineers agree on actionable events, response time, missed-event tolerance, and alert burden. The planner defines the production target, horizon, and planning baseline.</p></div><div><h3>2. Validate the evidence</h3><p>Data engineering supplies time-aligned historian, train-status, and work-order data with quality checks. Data science evaluates frozen candidates on later periods and separate trains.</p></div><div><h3>3. Run alongside operations</h3><p>Engineers review candidate alerts and forecasts without changing controls. Record useful alerts, missed events, response times, and planning error before deciding whether to expand.</p></div></div>
<details><summary>Review owners, acceptance criteria, monitoring, and fallback</summary>
<div class="table-wrap"><table><thead><tr><th>Stage / owner</th><th>Required evidence</th><th>Decision gate</th></tr></thead><tbody>
<tr><td>Scope / operations lead + planner</td><td>Event labels, forecast horizon, cost of misses, acceptable delay and alert burden</td><td>Owners approve numeric targets before model selection; the dashboard’s 30-minute window is only illustrative.</td></tr>
<tr><td>Data readiness / data engineer</td><td>Tag units, train IDs, timestamps, available-at times, quality flags, maintenance labels</td><td>Resolve critical missing data and time alignment; agree on freshness limits before evaluation.</td></tr>
<tr><td>Offline review / data scientist + reliability engineer</td><td>Future-period and separate-train tests; event misses, delays, false alerts, forecast error</td><td>Meet agreed detection requirements, lower alert burden without unacceptable delay, and improve forecast error against the planning baseline.</td></tr>
<tr><td>Shadow pilot / operations owner</td><td>Reviewed alerts, model versions, operator feedback, and a representative range of operating modes</td><td>Owner approves expansion only after acceptance criteria hold; otherwise revise or stop the candidate.</td></tr>
<tr><td>Ongoing support / data science + IT</td><td>Input freshness and missingness, mode changes, drift, alert rate, delayed labels, forecast error</td><td>Route breaches to the owner; review before retraining or releasing a new version.</td></tr>
</tbody></table></div><p><b>Fallback:</b> retain approved plant alerts and the established planning process. Suppress candidate output when critical inputs are stale or missing; identify its last successful run. Version data, features, thresholds, and evaluations so changes can be reviewed and rolled back. A model score never changes a control setpoint.</p>
<p><b>Dependencies:</b> historian access, agreed event labels, maintenance and schedule history, and engineering review time. Pilot duration and staffing would be agreed after those dependencies and event frequency are understood.</p></details>
<details><summary>Review the workflow, sources, and reproducibility</summary><p><b>Implemented:</b> Python ingestion → pandas validation and DuckDB SQL → baseline / MLP / XGBoost → saved predictions and evaluation → this static dashboard. Optional PySpark preparation demonstrates a path to larger data volumes. GitHub Actions rebuilds and publishes after checks pass.</p><p><b>Possible plant deployment:</b> Data Factory ingestion into Azure Data Lake; Databricks for preparation and scheduled scoring; governed outputs for engineering review. This is a proposed adaptation, not a deployed integration.</p><p><b>Sources:</b> <a href="https://github.com/camaramm/tennessee-eastman-profBraatz">Tennessee Eastman process simulation</a> and <a href="https://www.eia.gov/dnav/ng/hist_xls/N9133US2m.xls">EIA monthly U.S. LNG exports</a>. The two datasets answer separate questions and are not joined as if they represented the same facility.</p><p>Rebuild locally with <code>python download.py</code>, <code>python prepare.py</code>, <code>python model.py</code>, and <code>python dashboard.py</code>. Read the project README for evaluation design and limitations.</p></details></section>
<footer>Public-data portfolio · Python / SQL / scikit-learn / XGBoost · All displayed metrics are calculated from saved test predictions.</footer>
</main><script>document.querySelectorAll('details').forEach(d=>d.addEventListener('toggle',()=>{{if(d.open)d.querySelectorAll('.plotly-graph-div').forEach(g=>Plotly.Plots.resize(g));}}));</script></body></html>'''
    SITE.mkdir(exist_ok=True)
    (SITE / "plotly.min.js").write_text(get_plotlyjs())
    (SITE / "index.html").write_text(html)
    print(f"Built {SITE / 'index.html'}")


if __name__ == "__main__":
    main()
