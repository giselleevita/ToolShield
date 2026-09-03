const fmt = (value) => value == null ? "Not reported" : `${(value * 100).toFixed(1)}%`;
fetch("results.json").then(r => r.json()).then(report => {
  const protocols = [...new Set(report.results.map(r => r.protocol))];
  const models = [...new Set(report.results.map(r => r.model))];
  const protocol = document.querySelector("#protocol");
  const model = document.querySelector("#model");
  protocols.forEach(v => protocol.add(new Option(v, v)));
  models.forEach(v => model.add(new Option(v, v)));
  const render = () => {
    const row = report.results.find(r => r.protocol === protocol.value && r.model === model.value);
    const fields = row ? [
      ["ROC-AUC", fmt(row.roc_auc_mean)], ["PR-AUC", fmt(row.pr_auc_mean)],
      ["FPR at TPR 90", fmt(row.fpr_at_tpr_90_mean)], ["Median latency", row.latency_p50_ms_mean == null ? "Not reported" : `${row.latency_p50_ms_mean.toFixed(1)} ms`],
      ["Reported runs", String(row.n_seeds)], ["Seed status", row.seed_status]
    ] : [];
    document.querySelector("#metrics").innerHTML = fields.map(([k,v]) => `<div class="card"><span class="value">${v}</span>${k}</div>`).join("");
  };
  protocol.onchange = render; model.onchange = render; render();
});
