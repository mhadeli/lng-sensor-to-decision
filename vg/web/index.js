const H = DATA.headlines;
const byTerminal = Object.fromEntries(DATA.terminals.map(t => [t.terminal, t]));
const change = DATA.total_bcf / DATA.year_ago_bcf - 1;
setText('cutoff', `Reported through ${monthName(DATA.latest_month)}`);
setText('total', bcf(DATA.total_bcf));
setText('total-change', `${pct(Math.abs(change))} ${change >= 0 ? 'higher' : 'lower'} than a year earlier`);
setText('cargoes', DATA.total_cargoes);
setText('util-cp', pct(byTerminal['Calcasieu Pass'].utilization));
setText('util-pq', pct(byTerminal['Plaquemines'].utilization));

const months = DATA.monthly.map(r => r.month);
plot('overview-chart', ['Calcasieu Pass', 'Plaquemines'].map(t => ({
  x:months, y:DATA.monthly.map(r => r[t]), type:'bar', name:t, marker:{color:TERMINAL_COLORS[t]},
  hovertemplate:`${t}: %{y:.1f} Bcf<extra></extra>`})), {barmode:'stack', yaxis:{title:{text:'Exports (Bcf)'}}});

const ramp = Object.fromEntries(H.ramp.map(r => [r.terminal, r]));
const hh4 = Object.fromEntries(H.hh.map(r => [r.method, r]));
setText('h-exports', 'Destination mix by terminal and month, plus a Calcasieu Pass estimate for the first unreported month with its 12-month track record.');
setText('h-production', `Plaquemines reached 90% of nameplate in ${ramp['Plaquemines'].months_to_90pct} months (Calcasieu Pass: ${ramp['Calcasieu Pass'].months_to_90pct}). Each 1 °C warmer month costs Calcasieu Pass about ${num(Math.abs(H.heat.pct_nameplate_per_c), 1)}% of nameplate output.`);
setText('h-reliability', `${H.reliability.windows} low-output windows detected. ${pct(H.reliability.weather_match_rate)} coincide with a weather hazard, against ${pct(H.reliability.chance_match_rate)} expected by chance.`);
setText('h-commercial', `Months with a larger modeled Europe advantage sent more cargoes to Europe (rank correlation ${num(H.choice.spearman_rho, 2)}). The Conv1D + LSTM improves 4-week Henry Hub error by ${pct(hh4.conv_lstm.skill_vs_naive, 1)} over a naive forecast — not statistically significant.`);
setText('h-fleet', `${pct(H.asia_long_route_share)} of classifiable Asia round trips match Suez or Cape timings rather than Panama. AIS berth stays at Calcasieu Pass matched ${H.calls_matched} of ${H.calls} DOE departures in October 2023.`);
setText('h-data', `${H.sources} pinned sources with checksums, contract checks on every table, and DOE volumes within ${num(H.reconciliation_max_pct, 1)}% of what Venture Global reports to the SEC each quarter.`);
