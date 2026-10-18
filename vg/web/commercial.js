const NB = DATA.netbacks, C = DATA.choice;
const recent = NB.filter(r => r.month >= '2022-01');
const europeWins = recent.filter(r => r.netback_Europe > r['netback_Asia (via Panama)']).length;
setHTML('netback-answer', `Since January 2022, Europe paid more after shipping in <strong>${europeWins} of ${recent.length} months</strong> against Tokyo via Panama, and more often still when Panama is unavailable. Shipping to Asia costs roughly twice as much per MMBtu, so Asia needs a clear price premium to win.`);
const months = NB.map(r => r.month);
plot('netbacks', [
  {x:months, y:NB.map(r => r.netback_Europe), name:'Europe (Rotterdam)', mode:'lines', line:{color:REGION_COLORS.Europe, width:2.5}},
  {x:months, y:NB.map(r => r['netback_Asia (via Panama)']), name:'Asia via Panama', mode:'lines', line:{color:REGION_COLORS.Asia, width:2.5}},
  {x:months, y:NB.map(r => r['netback_Asia (no Panama)']), name:'Asia without Panama', mode:'lines', line:{color:REGION_COLORS.Asia, dash:'dot', width:1.5}},
  {x:months, y:NB.map(r => r.feed_gas_cost), name:'Feed gas (115% Henry Hub)', mode:'lines', line:{color:COLORS.grey, width:1.5}},
], {yaxis:{title:{text:'USD per MMBtu'}, tickprefix:'$'}});
plot('probability', [
  {x:months, y:NB.map(r => r.p_europe_beats_asia_panama), name:'P(Europe > Asia via Panama)', type:'bar', marker:{color:COLORS.teal}},
], {yaxis:{tickformat:'.0%', range:[0, 1], title:{text:'Probability'}}, legend:{orientation:'h', y:1.25}});

setHTML('choice-answer', `Yes, moderately. Across ${C.months} months, the bigger Europe's modeled advantage, the larger the share of Europe-plus-Asia volume sent to Europe (rank correlation <strong>${num(C.spearman_rho, 2)}</strong>, p = ${num(C.p_value, 3)}). The majority destination matched the better netback in <strong>${pct(C.direction_agreement)}</strong> of months.`);
const mix = DATA.mix;
plot('choice', [{x:mix.map(r => r.europe_minus_asia), y:mix.map(r => r.europe_share_of_europe_asia), mode:'markers', text:mix.map(r => r.month),
  marker:{size:9, color:mix.map(r => Number(r.month.slice(0, 4))), colorscale:[[0, '#c9ddd5'], [1, COLORS.ink]], showscale:true, colorbar:{title:{text:'Year'}, thickness:10}},
  hovertemplate:'%{text}: Europe +$%{x:.2f} → %{y:.0%} to Europe<extra></extra>'}],
  {hovermode:'closest', xaxis:{title:{text:'Modeled Europe netback minus Asia (USD/MMBtu)'}, zeroline:true, zerolinecolor:'#c5d2cc', showgrid:true, gridcolor:'#edf1ed'},
   yaxis:{title:{text:'Share of Europe + Asia volume to Europe'}, tickformat:'.0%', range:[0, 1.05]}});

const R = DATA.realized;
const gap = (t, from, to, f = v => v) => { const s = R.filter(r => r.terminal === t && r.quarter >= from && r.quarter <= to && r.modeled_netback !== null);
  return s.reduce((a, r) => a + f(r.realized_minus_modeled), 0) / s.length; };
setHTML('realized-answer', `Two regimes show up. Plaquemines, still selling commissioning cargoes at market prices, realized within <strong>${usd(gap('Plaquemines', '2025Q1', '2025Q4', Math.abs))}</strong> per MMBtu of the modeled market netback on average (absolute gap) through 2025, before falling below it in 2026, when a large share was sold to its affiliate VG Commodities (140.1 of 342.1 TBtu in Q2 2026, per the 10-Q) at an internal price. Calcasieu Pass fell an average <strong>${usd(-gap('Calcasieu Pass', '2025Q2', '2026Q4'))}</strong> below it after commercial operation in April 2025, when its long-term contracts at fixed fees plus 115% of Henry Hub began.`);
plot('realized', ['Calcasieu Pass', 'Plaquemines'].flatMap(t => {
  const s = R.filter(r => r.terminal === t);
  return [{x:s.map(r => r.quarter), y:s.map(r => r.price_per_mmbtu), name:`${t} realized (SEC)`, mode:'lines+markers', line:{color:TERMINAL_COLORS[t], width:2.5}},
          {x:s.map(r => r.quarter), y:s.map(r => r.modeled_netback), name:`${t} modeled netback`, mode:'lines', line:{color:TERMINAL_COLORS[t], dash:'dot', width:1.5}}];
}), {yaxis:{title:{text:'USD per MMBtu'}, tickprefix:'$', rangemode:'tozero'}, shapes:[{type:'line', xref:'x', yref:'paper', x0:'2025Q2', x1:'2025Q2', y0:0, y1:1, line:{color:COLORS.grey, dash:'dash', width:1}}],
   annotations:[{x:'2025Q2', y:1, yref:'paper', text:'Calcasieu Pass COD', showarrow:false, yanchor:'bottom', font:{size:11}}]});

const M = DATA.hh_metrics, m4 = Object.fromEntries(M.filter(r => r.horizon_weeks === 4).map(r => [r.method, r]));
setHTML('hh-answer', `Barely. Four weeks ahead the Conv1D + LSTM's average error is <strong>${usd(m4.conv_lstm.mae)}</strong> against <strong>${usd(m4.naive.mae)}</strong> for "price stays where it is" — a ${pct(m4.conv_lstm.skill_vs_naive, 1)} improvement that is not statistically significant (p = ${num(m4.conv_lstm.dm_p_value, 2)}). Adding last year's seasonal change makes forecasts clearly worse. For planning, Henry Hub behaves close to a random walk at these horizons.`);
const weekly = DATA.hh_weekly, test = DATA.hh_test;
plot('hh', [
  {x:weekly.map(r => r.week), y:weekly.map(r => r.price), name:'Weekly Henry Hub', mode:'lines', line:{color:COLORS.ink, width:1.5}},
  {x:test.map(r => r.target_week), y:test.map(r => r.conv_lstm), name:'Conv1D + LSTM, 4 weeks earlier', mode:'lines', line:{color:COLORS.teal, width:1.5}},
  {x:test.map(r => r.target_week), y:test.map(r => r.naive), name:'Naive, 4 weeks earlier', mode:'lines', line:{color:COLORS.amber, width:1, dash:'dot'}},
], {yaxis:{title:{text:'USD per MMBtu'}, tickprefix:'$'}});
table('hh-metrics', M, [
  {key:'horizon_weeks', label:'Weeks ahead', num:true}, {key:'label', label:'Method'},
  {key:'mae', label:'Avg error', num:true, format:v => usd(v)}, {key:'skill_vs_naive', label:'vs naive', num:true, format:v => pct(v, 1)},
  {key:'dm_p_value', label:'p-value', num:true, format:(v, r) => r.method === 'naive' ? '—' : num(v, 2)}]);
table('hh-outlook', DATA.hh_outlook, [
  {key:'target_week', label:'Week ending', format:day}, {key:'naive', label:'Naive', num:true, format:v => usd(v)},
  {key:'ridge', label:'Linear', num:true, format:v => usd(v)}, {key:'conv_lstm', label:'Conv1D + LSTM', num:true, format:v => usd(v)}]);
table('assumptions', DATA.assumptions, [
  {key:'parameter', label:'Parameter'}, {key:'low', label:'Low', num:true, format:v => num(v, 2)}, {key:'base', label:'Base', num:true, format:v => num(v, 2)},
  {key:'high', label:'High', num:true, format:v => num(v, 2)}, {key:'unit', label:'Unit'}, {key:'note', label:'Note'}]);
