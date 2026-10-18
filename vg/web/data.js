const silverTables = DATA.lineage.filter(r => r.layer === 'silver'), goldTables = DATA.lineage.filter(r => r.layer === 'gold');
setHTML('lineage-answer', `<strong>${DATA.sources.length} bronze sources</strong> → <strong>${silverTables.length} silver tables</strong> → <strong>${goldTables.length} gold marts</strong> → 7 dashboard pages, all in one local DuckDB warehouse (<code>data/warehouse.duckdb</code>).`);
const layer = (title, text, items) => `<div class="card" style="margin:0"><div class="eyebrow">${title}</div><p>${text}</p><p class="legend-note">${items}</p></div>`;
setHTML('layers', [
  layer('Bronze', 'Raw public files, pinned by checksum in <code>bronze/</code>.', DATA.sources.map(s => esc(s.id)).join(', ')),
  layer('Silver', 'Validated, typed tables keyed by terminal, vessel and date.', silverTables.map(t => esc(t.name)).join(', ')),
  layer('Gold', 'Analysis marts behind each question.', goldTables.map(t => esc(t.name)).join(', ')),
  layer('Site', 'Static pages with embedded data and CSV downloads.', 'Overview, Exports, Production, Reliability, Commercial, Fleet, Data'),
].join(''));

const R = DATA.reconciliation;
const worst = Math.max(...R.map(r => Math.abs(r.tbtu_difference_pct)));
const counted = R.filter(r => r.cargoes_exported !== null), exact = counted.filter(r => r.cargo_difference === 0).length;
setHTML('recon-answer', `Yes. Across ${R.length} terminal-quarters, DOE volumes are within <strong>${num(worst, 1)}%</strong> of the SEC figures, and cargo counts match exactly in <strong>${exact} of ${counted.length}</strong> quarters where the filing reports them.`);
plot('recon', ['Calcasieu Pass', 'Plaquemines'].map(t => {
  const s = R.filter(r => r.terminal === t);
  return {x:s.map(r => r.quarter), y:s.map(r => r.tbtu_difference_pct / 100), type:'bar', name:t, marker:{color:TERMINAL_COLORS[t]},
    hovertemplate:`${t}: %{y:.1%}<extra></extra>`};
}), {barmode:'group', yaxis:{tickformat:'.1%', title:{text:'DOE minus SEC'}}});
table('recon-table', R, [
  {key:'terminal', label:'Terminal'}, {key:'quarter', label:'Quarter'}, {key:'sec_tbtu', label:'SEC TBtu', num:true, format:v => num(v, 1)},
  {key:'doe_tbtu', label:'DOE TBtu', num:true, format:v => num(v, 1)}, {key:'tbtu_difference_pct', label:'Difference', num:true, format:v => `${num(v, 1)}%`},
  {key:'cargoes_exported', label:'SEC cargoes', num:true, format:v => v === null ? '—' : v}, {key:'doe_cargoes', label:'DOE cargoes', num:true},
  {key:'filing', label:'Filing', format:v => `<code>${esc(v)}</code>`}]);

const Q = DATA.quality, warnings = Q.filter(q => q.status !== 'passed');
setHTML('quality-answer', `<strong>${Q.filter(q => q.status === 'passed').length} of ${Q.length}</strong> tables passed every contract (${Q.reduce((s, q) => s + q.checks, 0)} checks: required columns, nulls, unique keys, value ranges, row counts). Any failure stops the build before the site is replaced. ${warnings.length ? `Findings recorded without stopping: ${warnings.map(w => esc(w.detail)).join('; ')}.` : ''}`);
table('quality', Q, [
  {key:'table', label:'Table', format:v => `<code>${esc(v)}</code>`}, {key:'rows', label:'Rows', num:true, format:v => v === null ? '—' : num(v, 0)},
  {key:'checks', label:'Checks', num:true}, {key:'status', label:'Status', format:v => v === 'passed' ? tag('Passed') : tag(v, 'warn')},
  {key:'detail', label:'Detail', format:v => esc(v || '')}]);
table('sources', DATA.sources, [
  {key:'id', label:'Source', format:(v, r) => r.url ? `<a href="${esc(Array.isArray(r.url) ? r.url[0] : r.url)}" target="_blank" rel="noopener">${esc(v)}</a>` : esc(v)},
  {key:'description', label:'Content'}, {key:'license', label:'License'},
  {key:'bytes', label:'Size', num:true, format:v => `${num(v / 1024, 0)} KB`},
  {key:'sha256', label:'SHA-256', format:v => `<code>${esc(v.slice(0, 12))}…</code>`},
  {key:'retrieved_at', label:'Retrieved', format:(v, r) => v ? day(v) : esc(r.builder || '')}]);
table('events', DATA.events, [
  {key:'date', label:'Date', format:(v, r) => r.precision === 'month' ? shortMonth(v) : day(v)}, {key:'terminal', label:'Terminal'},
  {key:'category', label:'Type'}, {key:'title', label:'Event'},
  {key:'source_name', label:'Source', format:(v, r) => `<a href="${esc(r.source_url)}" target="_blank" rel="noopener">${esc(v)}</a>`}]);
table('timings', DATA.timings, [{key:'step', label:'Step'}, {key:'seconds', label:'Seconds', num:true, format:v => num(v, 1)}]);
