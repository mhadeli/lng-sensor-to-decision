const share = DATA.share;
const months = unique(share.map(r => r.month)).sort();
const totals = months.map(m => share.filter(r => r.month === m).reduce((a, r) => ({c:a.c + r.cargoes, o:a.o + r.own}), {c:0, o:0}));
const last12 = totals.slice(-12).reduce((a, t) => ({c:a.c + t.c, o:a.o + t.o}), {c:0, o:0});
const firstOwn = months[totals.findIndex(t => t.o > 0)];
setHTML('share-answer', `Venture Global's own ships first loaded in ${monthName(firstOwn)} and carried <strong>${pct(last12.o / last12.c)}</strong> of cargoes over the last 12 reported months (${last12.o} of ${last12.c}). Most cargoes still sail on chartered or buyer-provided ships.`);
plot('share', [{x:months, y:totals.map(t => t.o / t.c), type:'bar', marker:{color:COLORS.teal}, name:'Own ships',
  customdata:totals.map(t => `${t.o} of ${t.c}`), hovertemplate:'%{y:.0%} (%{customdata} cargoes)<extra></extra>'}],
  {yaxis:{tickformat:'.0%', range:[0, 1], title:{text:'Share of cargoes'}}, showlegend:false});
table('ships', DATA.ships.slice(0, 15), [
  {key:'vessel', label:'Ship'}, {key:'own_fleet', label:'Owner', format:v => v ? tag('Venture Global') : tag('Third party', 'muted')},
  {key:'cargoes', label:'Cargoes', num:true}, {key:'volume_bcf', label:'Volume', num:true, format:v => bcf(v)},
  {key:'first', label:'First', format:day}, {key:'last', label:'Latest', format:day}]);

const trips = DATA.trips, regions = Object.keys(REGION_COLORS);
const median = a => { const s = [...a].sort((x, y) => x - y); return s.length ? s[Math.floor(s.length / 2)] : null; };
const med = Object.fromEntries(regions.map(r => [r, median(trips.filter(t => t.region === r).map(t => t.round_trip_days))]));
const asiaExpected = median(trips.filter(t => t.region === 'Asia').map(t => t.expected_shortest));
const routeCounts = DATA.routes.reduce((a, r) => ({...a, [r.route_inferred]:(a[r.route_inferred] || 0) + r.voyages}), {});
const classified = (routeCounts['Panama'] || 0) + (routeCounts['Suez or Cape'] || 0);
setHTML('trip-answer', `Median round trip: Europe <strong>${med.Europe} days</strong>, Latin America ${med['Latin America']}, Middle East &amp; Africa ${med['Middle East & Africa']}, Asia <strong>${med.Asia} days</strong>. Asia trips take far longer than the ~${Math.round(asiaExpected)} days a Panama transit implies: of ${classified} Asia voyages that fit only one route, <strong>${routeCounts['Suez or Cape'] || 0}</strong> match Suez or Cape timings and ${routeCounts['Panama'] || 0} match Panama. The pattern holds across the whole period, not only during the 2023–24 Panama drought: canal slot costs and queues make the longer routes the norm for these cargoes.`);
plot('trips', regions.map(r => ({y:trips.filter(t => t.region === r).map(t => t.round_trip_days), type:'box', name:r, marker:{color:REGION_COLORS[r]}, boxpoints:'outliers'})),
  {hovermode:'closest', showlegend:false, yaxis:{title:{text:'Days until next U.S. loading'}}});
const quarters = unique(DATA.routes.map(r => r.quarter)).sort();
plot('routes', ['Panama', 'Ambiguous', 'Suez or Cape'].map((route, i) => ({x:quarters, type:'bar', name:route,
  y:quarters.map(q => (DATA.routes.find(r => r.quarter === q && r.route_inferred === route) || {voyages:0}).voyages),
  marker:{color:[COLORS.teal, COLORS.grey, COLORS.blue][i]}})), {barmode:'stack', yaxis:{title:{text:'Asia voyages'}}});

const calls = DATA.calls, berth = calls.map(c => c.berth_hours), wait = calls.map(c => c.waiting_hours);
const matched = calls.filter(c => c.doe_departure).length, byDate = calls.filter(c => (c.match_method || '').startsWith('date')).length;
setHTML('calls-answer', `${calls.length} berth stays in October 2023. A typical loading took <strong>${num(median(berth), 0)} hours</strong> at the berth (range ${num(Math.min(...berth), 0)}–${num(Math.max(...berth), 0)}). Most ships berthed on arrival, but ${calls.filter(c => c.waiting_hours > 12).length} waited more than half a day — up to ${num(Math.max(...wait), 0)} hours. Every stay matched a DOE departure (${matched} of ${calls.length}; ${byDate} by date only because the AIS name differed, e.g. a renamed ship).`);
const order = [...calls].sort((a, b) => a.berth_start.localeCompare(b.berth_start));
plot('calls-chart', [
  {x:order.map(c => c.vessel), y:order.map(c => c.waiting_hours), type:'bar', name:'Waiting', marker:{color:COLORS.amber}},
  {x:order.map(c => c.vessel), y:order.map(c => c.berth_hours), type:'bar', name:'At berth', marker:{color:COLORS.teal}},
], {barmode:'stack', hovermode:'closest', yaxis:{title:{text:'Hours'}}, xaxis:{tickangle:-35, automargin:true}});
table('calls', order, [
  {key:'vessel', label:'AIS name'}, {key:'berth_start', label:'Berthed', format:v => v.replace('T', ' ').slice(0, 16)},
  {key:'berth_hours', label:'Berth h', num:true, format:v => num(v, 0)}, {key:'waiting_hours', label:'Waiting h', num:true, format:v => num(v, 0)},
  {key:'doe_departure', label:'DOE departure', format:day}, {key:'doe_vessel', label:'DOE name'},
  {key:'match_method', label:'Match', format:v => v.startsWith('date') ? tag(v, 'warn') : tag(v)}]);

const sp = DATA.speed;
setHTML('speed-answer', `Not measurably in this month. Across ${num(sp.points, 0)} open-water positions from ${sp.ships} ships (waves up to ${num(sp.max_wave_m, 1)} m), each extra metre of wave height changed speed by <strong>${num(sp.knots_per_m_wave, 2)} knots</strong> (95% interval ${num(sp.ci_low, 1)} to ${num(sp.ci_high, 1)}) — indistinguishable from zero. Each extra 10 knots of wind is associated with ${num(sp.knots_per_10kn_wind, 2)} knots (interval ${num(sp.wind_ci_low, 1)} to ${num(sp.wind_ci_high, 1)}, p = ${num(sp.p_value_wind, 2)}). Average open-water speed was ${num(sp.mean_speed_kn, 1)} knots.`);
const bins = unique(DATA.speed_bins.map(b => b.wave_bin));
plot('speed', ['ballast', 'laden'].map(leg => ({x:bins, type:'bar', name:leg === 'laden' ? 'Laden (leaving)' : 'Ballast (arriving)',
  y:bins.map(b => (DATA.speed_bins.find(r => r.wave_bin === b && r.leg === leg) || {}).mean),
  customdata:bins.map(b => (DATA.speed_bins.find(r => r.wave_bin === b && r.leg === leg) || {}).size),
  marker:{color:leg === 'laden' ? COLORS.teal : COLORS.grey}, hovertemplate:'%{y:.1f} kn (%{customdata} positions)<extra></extra>'})),
  {barmode:'group', xaxis:{title:{text:'Wave height (m)'}}, yaxis:{title:{text:'Average speed (knots)'}}});

const costs = DATA.costs;
setText('cost-eu', usd(costs.find(c => c.route === 'Rotterdam').cost_base));
setText('cost-asia', usd(costs.find(c => c.route === 'Tokyo via Panama').cost_base));
table('costs', costs, [
  {key:'route', label:'From Calcasieu Pass to'}, {key:'nautical_miles', label:'Nautical miles', num:true, format:v => num(v, 0)},
  {key:'round_trip_days', label:'Round trip (days)', num:true, format:v => num(v, 0)},
  {key:'cost_low', label:'Low', num:true, format:v => usd(v)}, {key:'cost_base', label:'Base', num:true, format:v => usd(v)},
  {key:'cost_high', label:'High', num:true, format:v => usd(v)}]);
