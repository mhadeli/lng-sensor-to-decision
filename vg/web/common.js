/* Shared helpers. Every page reads its own validated payload embedded in the HTML; no network calls. */
const DATA = JSON.parse(document.getElementById('page-data').textContent);
const COLORS = {teal:'#277e6a', ink:'#173a3c', amber:'#b88135', blue:'#3f6fb5', red:'#b5523f', grey:'#9aa9a6', light:'rgba(39,126,106,.14)'};
const TERMINAL_COLORS = {'Calcasieu Pass':COLORS.teal, 'Plaquemines':COLORS.blue};
const REGION_COLORS = {'Europe':COLORS.teal, 'Asia':COLORS.blue, 'Latin America':COLORS.amber, 'Middle East & Africa':COLORS.red};

const $ = id => document.getElementById(id);
const setText = (id, value) => { const el = $(id); if (el) el.textContent = value; };
const setHTML = (id, value) => { const el = $(id); if (el) el.innerHTML = value; };
const num = (v, d = 1) => v === null || v === undefined || Number.isNaN(v) ? '—' : Number(v).toLocaleString('en-US', {minimumFractionDigits:d, maximumFractionDigits:d});
const pct = (v, d = 0) => v === null || v === undefined ? '—' : `${num(v * 100, d)}%`;
const usd = (v, d = 2) => v === null || v === undefined ? '—' : `$${num(v, d)}`;
const bcf = (v, d = 1) => `${num(v, d)} Bcf`;
const monthName = m => new Date(`${m.slice(0, 7)}-01T00:00:00Z`).toLocaleDateString('en-US', {month:'long', year:'numeric', timeZone:'UTC'});
const shortMonth = m => new Date(`${m.slice(0, 7)}-01T00:00:00Z`).toLocaleDateString('en-US', {month:'short', year:'numeric', timeZone:'UTC'});
const day = d => d ? new Date(`${d.slice(0, 10)}T00:00:00Z`).toLocaleDateString('en-US', {day:'numeric', month:'short', year:'numeric', timeZone:'UTC'}) : '—';
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const unique = values => [...new Set(values)];

function layout(extra = {}) {
  const base = {font:{family:'system-ui, sans-serif', color:'#526968', size:12}, paper_bgcolor:'transparent', plot_bgcolor:'transparent',
    margin:{l:56, r:18, t:16, b:44}, hovermode:'x unified', xaxis:{showgrid:false}, yaxis:{gridcolor:'#edf1ed', zeroline:false},
    legend:{orientation:'h', y:1.14, x:0}};
  return {...base, ...extra, xaxis:{...base.xaxis, ...(extra.xaxis || {})}, yaxis:{...base.yaxis, ...(extra.yaxis || {})}};
}
const CONFIG = {responsive:true, displayModeBar:false};
const plot = (id, traces, extra) => Plotly.react(id, traces, layout(extra), CONFIG);

/* columns: [{key, label, format?, num?}] */
function table(id, rows, columns) {
  const head = columns.map(c => `<th class="${c.num ? 'num' : ''}">${esc(c.label)}</th>`).join('');
  const body = rows.map(r => `<tr>${columns.map(c => {
    const value = c.format ? c.format(r[c.key], r) : esc(r[c.key]);
    return `<td class="${c.num ? 'num' : ''}">${value}</td>`;
  }).join('')}</tr>`).join('');
  setHTML(id, `<div class="table-wrap"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`);
}

function tag(text, kind = '') { return `<span class="tag ${kind}">${esc(text)}</span>`; }

setText('footer-cutoff', DATA.cutoff ? monthName(DATA.cutoff) : '');
setText('footer-built', DATA.built_at ? new Date(DATA.built_at).toLocaleDateString('en-US', {day:'numeric', month:'short', year:'numeric'}) : '');
