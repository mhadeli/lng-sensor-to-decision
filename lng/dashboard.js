/* Every chart is derived from the same embedded, validated dataset; no network calls. */
const data = JSON.parse(document.getElementById('dashboard-data').textContent);
const monthName = value => new Date(`${value}-01T00:00:00Z`).toLocaleDateString('en-US', {month:'long', year:'numeric', timeZone:'UTC'});
const setText = (id, value) => { document.getElementById(id).textContent = value; };
const bcf = value => `${value.toFixed(1)} Bcf`;
const latest = data.monthly.at(-1), previous = data.monthly.at(-2), forecast = data.forecast;
const change = (latest.volume_bcf / previous.volume_bcf - 1) * 100;
setText('cutoff', `Reported through ${monthName(latest.month)}`);
setText('volume', bcf(latest.volume_bcf));
setText('change', `${Math.abs(change).toFixed(1)}% ${change >= 0 ? 'higher' : 'lower'} than ${monthName(previous.month)}`);
setText('ships', latest.shipments); setText('countries', latest.destinations);
setText('error', bcf(forecast.mae_bcf)); setText('target', monthName(forecast.target_month));
setText('estimate', bcf(forecast.estimate_bcf));
setText('range', `Estimated range: ${bcf(forecast.lower_bcf)} – ${bcf(forecast.upper_bcf)}`);
setText('method', `Based on: ${forecast.method_label.toLowerCase()}, adjusted for the number of days in the month.`);
setText('coverage', `In the 12 test months, ${Math.round(forecast.interval_coverage * 12)} outcomes fell inside the range.`);
setText('test-caption', `${monthName(forecast.test_first)} – ${monthName(forecast.test_last)} · Selected method error: ${bcf(forecast.mae_bcf)} · Latest-month baseline error: ${bcf(forecast.baseline_mae_bcf)}.`);
setText('built', `Built ${new Date(data.source.built_at).toLocaleDateString('en-US')}`);
document.getElementById('source-link').href = data.source.url;
const layout = {font:{family:'system-ui, sans-serif',color:'#526968',size:12}, paper_bgcolor:'transparent',plot_bgcolor:'transparent',margin:{l:50,r:20,t:20,b:45},hovermode:'x unified',yaxis:{title:{text:'Export volume (Bcf)'},gridcolor:'#edf1ed',rangemode:'tozero'},xaxis:{showgrid:false},legend:{orientation:'h',y:1.16}};
const config = {responsive:true,displayModeBar:false};
function drawOverview() {
  const period = document.getElementById('period').value;
  const rows = period === 'all' ? data.monthly : data.monthly.slice(-Number(period));
  Plotly.react('trend', [{x:rows.map(r=>r.month), y:rows.map(r=>r.volume_bcf),type:'bar',name:'Exports',marker:{color:rows.map(r=>r.review_flag?'#b88135':'#277e6a')},hovertemplate:'%{y:.1f} Bcf<extra></extra>'}],layout,config);
  const flagged = rows.filter(r=>r.review_flag);
  setText('insight', flagged.length ? `${flagged.length} month(s) in this view were more than 25% below the previous six-month average (shown in amber). Review shipment timing and public operational updates for context.` : 'No months in this view were more than 25% below the previous six-month average. This is a shipment-volume check, not an equipment health assessment.');
  const months = new Set(rows.map(r=>r.month)); const totals = {};
  data.destinations.filter(r=>months.has(r.month)).forEach(r=>{totals[r.destination]=(totals[r.destination]||0)+r.volume_bcf;});
  const ranked = Object.entries(totals).sort((a,b)=>b[1]-a[1]).slice(0,6).reverse();
  Plotly.react('destinations',[{x:ranked.map(r=>r[1]),y:ranked.map(r=>r[0]),type:'bar',orientation:'h',marker:{color:'#277e6a'},hovertemplate:'%{x:.1f} Bcf<extra></extra>'}],{...layout,hovermode:'closest',margin:{l:115,r:15,t:15,b:45},xaxis:{title:{text:'Export volume (Bcf)'},gridcolor:'#edf1ed'},yaxis:{automargin:true}},config);
}
const test = data.test;
Plotly.newPlot('backtest',[
  {x:test.map(r=>r.month),y:test.map(r=>r.lower),mode:'lines',line:{width:0},showlegend:false,hoverinfo:'skip'},
  {x:test.map(r=>r.month),y:test.map(r=>r.upper),mode:'lines',line:{width:0},fill:'tonexty',fillcolor:'rgba(39,126,106,.12)',name:'Estimated range',hoverinfo:'skip'},
  {x:test.map(r=>r.month),y:test.map(r=>r.actual),mode:'lines+markers',name:'Reported exports',line:{color:'#173a3c',width:3}},
  {x:test.map(r=>r.month),y:test.map(r=>r.prediction),mode:'lines',name:'Estimate',line:{color:'#277e6a',dash:'dot',width:3}}
],layout,config);
document.getElementById('period').addEventListener('change',drawOverview);
drawOverview();
