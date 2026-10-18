"""Render the static, offline dashboard: one page per question area, with embedded data and CSV downloads."""
import json
import shutil
from pathlib import Path

from plotly.offline import get_plotlyjs

from vg.common import ROOT, SITE

WEB = Path(__file__).parent / "web"
PAGES = [("index", "Overview"), ("exports", "Exports"), ("production", "Production"), ("reliability", "Reliability"),
         ("commercial", "Commercial"), ("fleet", "Fleet"), ("data", "Data & methods")]


def nav(current):
    return "".join(f'<a href="{name}.html"{" aria-current=page" if name == current else ""}>{label}</a>' for name, label in PAGES)


def render(payloads, downloads):
    staging = SITE.with_name("site.tmp")
    shutil.rmtree(staging, ignore_errors=True)
    (staging / "assets").mkdir(parents=True)
    (staging / "data").mkdir()
    (staging / "docs").mkdir()
    base = (WEB / "base.html").read_text()
    for name, label in PAGES:
        data = json.dumps(payloads[name], allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
        html = (base.replace("{{title}}", label).replace("{{nav}}", nav(name)).replace("{{script}}", f"{name}.js")
                .replace("{{content}}", (WEB / f"{name}.html").read_text()).replace("{{data}}", data))
        (staging / f"{name}.html").write_text(html)
        shutil.copyfile(WEB / f"{name}.js", staging / "assets" / f"{name}.js")
    for asset in ["style.css", "common.js"]:
        shutil.copyfile(WEB / asset, staging / "assets" / asset)
    (staging / "assets" / "plotly.min.js").write_text(get_plotlyjs())
    for filename, frame in downloads.items():
        frame.to_csv(staging / "data" / filename, index=False)
    for doc in (ROOT / "docs").glob("*.md"):
        shutil.copyfile(doc, staging / "docs" / doc.name)
    # Swap in the finished site only after every page rendered.
    shutil.rmtree(SITE, ignore_errors=True)
    staging.rename(SITE)
