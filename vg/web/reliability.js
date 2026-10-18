const S = DATA.stats, cfg = DATA.settings;
const kindTag = k => k === 'weather' ? tag('Weather', '') : k === 'context' ? tag('Documented event', 'warn') : tag('Unexplained — review', 'bad');
setText('settings', `A cargo gap is flagged when no ship leaves for at least ${cfg.min_gap_days} days and ${cfg.gap_factor}× the recent typical gap. A volume dip is flagged when the 30-day pace falls below ${pct(cfg.dip_ratio)} of the previous 90 days. Overlapping flags merge into one window.`);
setText('warmup', cfg.warmup_days);

const lost = DATA.windows.reduce((s, w) => s + w.lost_bcf, 0);
const counts = k => DATA.windows.filter(w => w.explanation === k).length;
const where = ['Calcasieu Pass', 'Plaquemines'].map(t => `${t}: ${DATA.windows.filter(w => w.terminal === t).length}`).join(', ');
setHTML('detect-answer', `<strong>${S.windows} low-output windows</strong> (${where}), with an estimated shortfall of <strong>${bcf(lost, 0)}</strong> against the prior pace. ${counts('weather')} coincide with weather, ${counts('context')} with a documented event, and ${counts('unexplained')} have no public explanation.`);

function draw() {
  const t = $('terminal').value;
  const pace = DATA.pace.filter(r => r.terminal === t);
  const windows = DATA.windows.filter(w => w.terminal === t);
  const hazards = DATA.hazards.filter(h => h.terminal === t);
  const events = DATA.events.filter(e => e.terminal === t);
  const top = Math.max(...pace.map(r => r.pace30_bcfd)) * 1.1;
  const shapes = windows.map(w => ({type:'rect', xref:'x', yref:'paper', x0:w.start, x1:w.end, y0:0, y1:1, line:{width:0},
    fillcolor:w.explanation === 'unexplained' ? 'rgba(181,82,63,.14)' : 'rgba(184,129,53,.16)'}));
  plot('pace', [
    {x:pace.map(r => r.date), y:pace.map(r => r.pace30_bcfd), name:'30-day pace', mode:'lines', line:{color:TERMINAL_COLORS[t], width:2}},
    {x:pace.map(r => r.date), y:pace.map(r => r.prior90_bcfd), name:'Prior 90-day pace', mode:'lines', line:{color:COLORS.grey, dash:'dot', width:1.5}},
    {x:hazards.map(h => h.start), y:hazards.map(() => top * 0.97), mode:'markers', name:'Weather hazard', text:hazards.map(h => h.label),
      marker:{symbol:'triangle-down', size:9, color:hazards.map(h => h.followed_by_low_output ? COLORS.red : COLORS.blue)}, hovertemplate:'%{text}<extra></extra>'},
    {x:events.map(e => e.date), y:events.map(() => top * 0.05), mode:'markers', name:'Documented event', text:events.map(e => `${e.title} (${e.source_name})`),
      marker:{symbol:'diamond', size:9, color:COLORS.amber}, hovertemplate:'%{x|%b %Y}: %{text}<extra></extra>'},
  ], {shapes, yaxis:{title:{text:'Exports (Bcf/d)'}, range:[0, top]}, hovermode:'closest'});
}

table('windows-table', DATA.windows, [
  {key:'terminal', label:'Terminal'}, {key:'start', label:'Start', format:day}, {key:'end', label:'End', format:day},
  {key:'days', label:'Days', num:true}, {key:'detector', label:'Detector'},
  {key:'lost_bcf', label:'Shortfall', num:true, format:v => bcf(v)}, {key:'explanation', label:'Explanation', format:kindTag},
  {key:'matched', label:'Matched record', format:v => esc(v || '—')}]);

setHTML('weather-answer', `Partly. <strong>${pct(S.weather_match_rate)}</strong> of windows began during a weather hazard, compared with <strong>${pct(S.chance_match_rate)}</strong> expected by chance — a hint, not proof, with only ${S.windows} windows. In the other direction, only <strong>${S.hazards_followed_by_low_output} of ${S.weather_hazards}</strong> weather hazards since start-up were followed by a measurable drop: most storms, gales and freezes did not interrupt exports.`);
const weather = DATA.hazards.filter(h => h.hazard === 'tropical' || h.hazard === 'cold' || h.followed_by_low_output);
table('hazards-table', weather, [
  {key:'terminal', label:'Terminal'}, {key:'start', label:'Window from', format:day}, {key:'hazard', label:'Type'},
  {key:'label', label:'Hazard'}, {key:'followed_by_low_output', label:'Low output?', format:v => v ? tag('Yes', 'bad') : tag('No', 'muted')}]);

$('terminal').addEventListener('change', draw);
draw();
