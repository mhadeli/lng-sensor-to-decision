const terminals = ['Calcasieu Pass', 'Plaquemines'];
const info = Object.fromEntries(DATA.terminals.map(t => [t.terminal, t]));
const rampInfo = Object.fromEntries(DATA.ramp.map(r => [r.terminal, r]));
const series = t => DATA.utilization.filter(r => r.terminal === t);

const lastMonth = DATA.utilization.at(-1).month;
const recent = t => { const s = series(t).slice(-12); return s.reduce((a, r) => a + r.utilization, 0) / s.length; };
setHTML('util-answer', `Over the last 12 reported months Calcasieu Pass averaged <strong>${pct(recent('Calcasieu Pass'))}</strong> of nameplate and Plaquemines <strong>${pct(recent('Plaquemines'))}</strong>. Both run above the vendor-guaranteed nameplate, as Venture Global's filings describe.`);

const traces = [];
terminals.forEach(t => {
  const s = series(t);
  traces.push({x:s.map(r => r.month), y:s.map(r => r.utilization), name:t, mode:'lines', line:{color:TERMINAL_COLORS[t], width:2.5},
    customdata:s.map(r => r.bcfd), hovertemplate:`${t}: %{y:.0%} (%{customdata:.2f} Bcf/d)<extra></extra>`});
  const ratio = info[t].authorized_mtpa / info[t].nameplate_mtpa;
  traces.push({x:[s[0].month, lastMonth], y:[ratio, ratio], name:`${t} authorized`, mode:'lines', showlegend:false,
    line:{color:TERMINAL_COLORS[t], dash:'dash', width:1}, hovertemplate:`${t} authorized: %{y:.0%}<extra></extra>`});
});
traces.push({x:[DATA.utilization[0].month, lastMonth], y:[1, 1], name:'Nameplate', mode:'lines', line:{color:COLORS.grey, width:1.5}, hoverinfo:'skip'});
plot('util', traces, {yaxis:{tickformat:'.0%', title:{text:'Share of nameplate'}, rangemode:'tozero'}});

setHTML('ramp-answer', terminals.map(t => `${t} reached 90% of nameplate (three-month average) <strong>${rampInfo[t].months_to_90pct} months</strong> after its first cargo`).join('; ') +
  `. Plaquemines' three-month peak is ${pct(rampInfo['Plaquemines'].peak_3m_utilization)} of nameplate, against ${pct(rampInfo['Calcasieu Pass'].peak_3m_utilization)} for Calcasieu Pass.`);
plot('ramp', terminals.map(t => {
  const s = series(t).filter(r => r.utilization_3m !== null);
  return {x:s.map(r => r.months_since_first_cargo), y:s.map(r => r.utilization_3m), name:t, mode:'lines+markers',
    marker:{size:5}, line:{color:TERMINAL_COLORS[t], width:2.5}, customdata:s.map(r => r.month),
    hovertemplate:`${t}, %{customdata}: %{y:.0%}<extra></extra>`};
}), {hovermode:'closest', xaxis:{title:{text:'Months since first cargo'}}, yaxis:{tickformat:'.0%', title:{text:'Share of nameplate (3-month avg)'}}});

const all = DATA.heat.find(r => r.sample === 'All steady-state months');
const robust = DATA.heat.find(r => r.sample !== 'All steady-state months');
setHTML('heat-answer', `Yes, moderately. Each 1 °C warmer month is associated with <strong>${num(Math.abs(all.bcfd_per_c * 1000), 0)} MMcf/d less output</strong> — about <strong>${num(Math.abs(all.pct_nameplate_per_c), 1)}% of nameplate per °C</strong> (95% interval ${num(Math.abs(all.ci_high * 1000), 0)}–${num(Math.abs(all.ci_low * 1000), 0)} MMcf/d). A summer month around 10 °C warmer than winter therefore loses roughly ${num(Math.abs(all.pct_nameplate_per_c) * 10, 0)}% of nameplate, consistent with gas turbines losing power in hot air. Dropping months with detected low-output windows gives ${num(Math.abs(robust.pct_nameplate_per_c), 2)}% per °C (p = ${num(robust.p_value, 3)}), so the effect is not just storms or outages.`);
const pts = DATA.heat_points;
plot('heat', [false, true].map(after => {
  const s = pts.filter(r => Boolean(r.after_cod) === after);
  return {x:s.map(r => r.temp_mean_c), y:s.map(r => r.bcfd), mode:'markers', name:after ? 'After COD (Apr 2025)' : 'Before COD',
    marker:{size:9, color:after ? COLORS.teal : COLORS.amber, symbol:s.map(r => r.excluded ? 'circle-open' : 'circle')},
    customdata:s.map(r => r.month), hovertemplate:'%{customdata}: %{x:.1f} °C, %{y:.2f} Bcf/d<extra></extra>'};
}), {hovermode:'closest', xaxis:{title:{text:'Monthly mean temperature (°C)'}, showgrid:true, gridcolor:'#edf1ed'}, yaxis:{title:{text:'Export pace (Bcf/d)'}}});
table('heat-table', DATA.heat, [
  {key:'sample', label:'Sample'}, {key:'months', label:'Months', num:true},
  {key:'pct_nameplate_per_c', label:'% nameplate per °C', num:true, format:v => num(v, 2)},
  {key:'p_value', label:'p-value', num:true, format:v => num(v, 3)}, {key:'r_squared', label:'R²', num:true, format:v => num(v, 2)}]);
