"""Render a small, offline dashboard from the local workflow's saved artifacts."""
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from string import Template

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

from common import GOLD, NAMES, RAW, ROOT, SITE


def build_metadata():
    revision = os.environ.get("GITHUB_SHA")
    if not revision:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
        revision = commit.stdout.strip() if commit.returncode == 0 else "local"
        if status.stdout.strip():
            revision += "-working-tree"
    return {"revision": revision, "built_at": datetime.now(timezone.utc).isoformat(),
            "workflow": "local-maintenance-and-terminal-forecast"}


def table(rows):
    return '<div class="table-wrap" tabindex="0">' + pd.DataFrame(rows).to_html(index=False, border=0, escape=True) + '</div>'


def main():
    build = build_metadata()
    maintenance = json.loads((GOLD / "maintenance_evaluation.json").read_text())
    forecast = json.loads((GOLD / "forecast_evaluation.json").read_text())
    quality = json.loads((GOLD / "quality.json").read_text())
    replay = json.loads((GOLD / "replay_summary.json").read_text())
    selected = maintenance["selected"]
    unseen = maintenance["test"][selected]["unseen_machines"]
    comparison = []
    for kind, cohorts in maintenance["test"].items():
        for cohort, metric in cohorts.items():
            comparison.append({"Approach": NAMES[kind], "Test group": cohort.replace("_", " "),
                "Failures warned": f"{metric['warned_events']} / {metric['events']}",
                "Median lead": f"{metric['median_lead_hours']:.0f} h" if metric['median_lead_hours'] is not None else "—",
                "False reviews / 100 machine-days": f"{metric['false_alerts_per_100_machine_days']:.2f}",
                "Average precision": f"{metric['average_precision']:.3f}"})
    export_rows = [{"Approach": "Daily-rate baseline" if kind == "baseline" else NAMES[kind],
                    "Average error": f"{m['mae_bcf']:.2f} Bcf", "Percentage error": f"{m['mape']:.1%}"}
                   for kind, m in forecast["test"].items()]
    predictions = pd.read_csv(GOLD / "terminal_forecast.csv")
    figure = go.Figure()
    figure.add_scatter(x=predictions.month, y=predictions.actual / 1000, name="Actual exports",
                       line=dict(color="#213c3c", width=3))
    for kind, color in [("baseline", "#999f9d"), ("mlp", "#426baf"), ("xgboost", "#087f73")]:
        figure.add_scatter(x=predictions.month, y=predictions[kind] / 1000,
                          name="Daily-rate baseline" if kind == "baseline" else NAMES[kind],
                          line=dict(color=color, width=2, dash="dot" if kind == "baseline" else "solid"),
                          visible=True if kind in {"baseline", forecast['selected']} else "legendonly")
    figure.update_layout(template="plotly_white", height=340, margin=dict(l=45,r=15,t=20,b=65),
                         legend=dict(orientation="h",y=-.25), font=dict(family="Arial",color="#405051"),
                         hovermode="x unified", yaxis_title="Monthly exports (Bcf)")
    chart = figure.to_html(full_html=False, include_plotlyjs=False, div_id="terminal-chart",
                           config={"displayModeBar": False, "responsive": True})
    scores = pd.read_csv(GOLD / "replay_scores.csv")
    batches = []
    for timestamp, rows in scores.groupby("datetime", sort=True):
        rows = rows.sort_values("score", ascending=False, na_position="last")
        batches.append({"time": timestamp, "scored": int(rows.score.notna().sum()),
                        "suppressed": int(rows.score.isna().sum()), "reviews": int(rows.review_requested.sum()),
                        "rows": json.loads(rows[["machineID","score","status","review_requested"]].head(8).to_json(orient="records"))})
    selected_forecast = forecast["test"][forecast["selected"]]
    improvement = 1 - selected_forecast["mae_bcf"] / forecast["test"]["baseline"]["mae_bcf"]
    message = (f"{abs(improvement):.0%} {'less' if improvement >= 0 else 'more'} error than the daily-rate baseline on the test period."
               if forecast["selected"] != "baseline" else "The simple baseline won on validation; added complexity was not selected.")
    context = {"warned": unseen["warned_events"], "events": unseen["events"], "unseen_machines": unseen["machines"],
        "lead": f"{unseen['median_lead_hours']:.0f} h" if unseen['median_lead_hours'] is not None else "—",
        "false_rate": f"{unseen['false_alerts_per_100_machine_days']:.2f}", "chosen": NAMES[selected],
        "model_table": table(comparison), "export_table": table(export_rows), "forecast_chart": chart,
        "forecast_error": f"{selected_forecast['mae_bcf']:.2f}", "forecast_pct": f"{selected_forecast['mape']:.1%}",
        "forecast_selected": NAMES[forecast["selected"]], "forecast_message": message,
        "forecast_first": forecast["test_first"], "forecast_last": forecast["test_last"],
        "telemetry_rows": f"{quality['telemetry']['valid_rows']:,}", "feature_rows": f"{quality['features']['rows']:,}",
        "quarantined": sum(v.get("quarantined", 0) for v in quality.values() if isinstance(v,dict)),
        "replay_first": replay["first"][:10], "replay_last": replay["last"][:10],
        "replay_reviews": replay["reviews"], "replay_batches": replay["batches"],
        "threshold": f"{replay['threshold']:.2f}",
        "replay_data": json.dumps(batches, allow_nan=False).replace("<", "\\u003c"),
        "raw_rows": f"{quality['exports']['source_rows']:,}",
        "build_label": build["revision"][:7] + (" · local changes" if build["revision"].endswith("-working-tree") else "")}
    SITE.mkdir(exist_ok=True)
    for asset in ("dashboard.css", "dashboard.js"):
        shutil.copyfile(ROOT / "assets" / asset, SITE / asset)
    (SITE / "plotly.min.js").write_text(get_plotlyjs())
    (SITE / "index.html").write_text(Template((ROOT / "templates" / "dashboard.html").read_text()).substitute(context))
    (SITE / "build_metadata.json").write_text(json.dumps(build, indent=2) + "\n")
    pd.DataFrame(comparison).to_csv(SITE / "model_comparison.csv", index=False)
    shutil.copyfile(GOLD / "terminal_forecast.csv", SITE / "terminal_forecast.csv")
    shutil.copyfile(RAW / "manifest.json", SITE / "source_manifest.json")
    for document in (ROOT / "docs").glob("*.md"):
        shutil.copyfile(document, SITE / document.name)
    print(f"Open {SITE / 'index.html'}")


if __name__ == "__main__":
    main()
