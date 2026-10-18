const flags = Object.fromEntries(DATA.cp_flags.map(r => [r.month, r.review_flag]));
const allMonths = unique(DATA.monthly.map(r => r.month)).sort();

function selection() {
  const terminal = $('terminal').value, period = $('period').value;
  const months = period === 'all' ? allMonths : allMonths.slice(-Number(period));
  const terminals = terminal === 'all' ? ['Calcasieu Pass', 'Plaquemines'] : [terminal];
  return {months, terminals, monthSet:new Set(months)};
}

function draw() {
  const {months, terminals, monthSet} = selection();
  const rows = DATA.monthly.filter(r => monthSet.has(r.month) && terminals.includes(r.terminal));
  plot('trend', terminals.map(t => {
    const series = rows.filter(r => r.terminal === t);
    return {x:series.map(r => r.month), y:series.map(r => r.volume_bcf), type:'bar', name:t,
      marker:{color:series.map(r => t === 'Calcasieu Pass' && flags[r.month] ? COLORS.amber : TERMINAL_COLORS[t])},
      customdata:series.map(r => r.cargoes), hovertemplate:`${t}: %{y:.1f} Bcf · %{customdata} cargoes<extra></extra>`};
  }), {barmode:'stack', yaxis:{title:{text:'Exports (Bcf)'}}});

  const last = months.at(-1), first = months[0];
  const total = m => rows.filter(r => r.month === m).reduce((s, r) => s + r.volume_bcf, 0);
  const sum = rows.reduce((s, r) => s + r.volume_bcf, 0);
  setHTML('trend-answer', `<strong>${bcf(sum, 0)}</strong> exported from ${shortMonth(first)} to ${shortMonth(last)} (${terminals.join(' and ')}). The latest month, ${monthName(last)}, shipped <strong>${bcf(total(last))}</strong> in ${rows.filter(r => r.month === last).reduce((s, r) => s + r.cargoes, 0)} cargoes.`);

  const dest = DATA.destinations.filter(r => monthSet.has(r.month) && terminals.includes(r.terminal));
  const byCountry = {};
  dest.forEach(r => { byCountry[r.destination] = (byCountry[r.destination] || 0) + r.volume_bcf; });
  const top = Object.entries(byCountry).sort((a, b) => b[1] - a[1]).slice(0, 10).reverse();
  plot('countries', [{x:top.map(r => r[1]), y:top.map(r => r[0]), type:'bar', orientation:'h', marker:{color:COLORS.teal},
    hovertemplate:'%{y}: %{x:.1f} Bcf<extra></extra>'}],
    {hovermode:'closest', margin:{l:120, r:16, t:10, b:44}, xaxis:{title:{text:'Exports (Bcf)'}, gridcolor:'#edf1ed'}, yaxis:{automargin:true}});

  const regions = Object.keys(REGION_COLORS);
  const quarters = unique(months.map(m => `${m.slice(0, 4)} Q${Math.ceil(Number(m.slice(5, 7)) / 3)}`));
  const qOf = m => `${m.slice(0, 4)} Q${Math.ceil(Number(m.slice(5, 7)) / 3)}`;
  const totals = Object.fromEntries(quarters.map(q => [q, 0]));
  dest.forEach(r => { totals[qOf(r.month)] += r.volume_bcf; });
  plot('regions', regions.map(region => ({x:quarters, type:'bar', name:region, marker:{color:REGION_COLORS[region]},
    y:quarters.map(q => dest.filter(r => r.region === region && qOf(r.month) === q).reduce((s, r) => s + r.volume_bcf, 0) / totals[q]),
    hovertemplate:`${region}: %{y:.0%}<extra></extra>`})), {barmode:'stack', yaxis:{tickformat:'.0%', range:[0, 1]}});
}

const f = DATA.forecast, test = DATA.test;
setText('target', monthName(f.target_month));
setText('estimate', bcf(f.estimate_bcf));
setText('range', `Estimated range: ${bcf(f.lower_bcf)} – ${bcf(f.upper_bcf)}`);
setText('method', `Method chosen on earlier months: ${f.method_label.toLowerCase()}, adjusted for days in the month.`);
setText('coverage', `In the 12 test months, ${Math.round(f.interval_coverage * 12)} outcomes fell inside the range.`);
setText('test-caption', `${monthName(f.test_first)} – ${monthName(f.test_last)} · average error ${bcf(f.mae_bcf)} (latest-month baseline: ${bcf(f.baseline_mae_bcf)}).`);
plot('backtest', [
  {x:test.map(r => r.month), y:test.map(r => r.lower), mode:'lines', line:{width:0}, showlegend:false, hoverinfo:'skip'},
  {x:test.map(r => r.month), y:test.map(r => r.upper), mode:'lines', line:{width:0}, fill:'tonexty', fillcolor:COLORS.light, name:'Estimated range', hoverinfo:'skip'},
  {x:test.map(r => r.month), y:test.map(r => r.actual), mode:'lines+markers', name:'Reported', line:{color:COLORS.ink, width:3}},
  {x:test.map(r => r.month), y:test.map(r => r.prediction), mode:'lines', name:'Estimate', line:{color:COLORS.teal, dash:'dot', width:3}},
], {yaxis:{title:{text:'Exports (Bcf)'}}});

$('terminal').addEventListener('change', draw);
$('period').addEventListener('change', draw);
draw();
