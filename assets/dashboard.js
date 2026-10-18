const batches = JSON.parse(document.getElementById('replay-data').textContent);
const slider = document.getElementById('replay-index');
slider.max = batches.length - 1;
function showBatch() {
  const batch = batches[Number(slider.value)];
  document.getElementById('replay-time').textContent = batch.time;
  document.getElementById('replay-status').textContent =
    `${batch.scored} machines scored · ${batch.suppressed} suppressed · ${batch.reviews} new reviews`;
  const body = document.getElementById('replay-rows');
  body.replaceChildren();
  for (const row of batch.rows) {
    const tr = document.createElement('tr');
    const values = [row.machineID, row.score === null ? '—' : row.score.toFixed(3),
      row.status.startsWith('Suppressed') ? row.status : row.review_requested ? 'Review requested' : 'No new request'];
    for (const value of values) {
      const td = document.createElement('td');
      td.textContent = value;
      tr.appendChild(td);
    }
    body.appendChild(tr);
  }
}
slider.addEventListener('input', showBatch);
showBatch();
document.querySelectorAll('details').forEach(detail => detail.addEventListener('toggle', () => {
  if (detail.open) detail.querySelectorAll('.plotly-graph-div').forEach(graph => Plotly.Plots.resize(graph));
}));
