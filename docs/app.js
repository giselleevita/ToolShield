const fmt = (value) => value == null ? "Not reported" : `${(value * 100).toFixed(1)}%`;
const shortHash = (value) => `${value.slice(0, 12)}…`;
fetch("results.json").then(r => {
  if (!r.ok) throw new Error(`Results request failed (${r.status})`);
  return r.json();
}).then(report => {
  const protocols = [...new Set(report.results.map(r => r.protocol))];
  const protocol = document.querySelector("#protocol");
  const model = document.querySelector("#model");
  protocols.forEach(v => protocol.add(new Option(v, v)));
  const syncModels = () => {
    const selected = model.value;
    const models = report.results.filter(r => r.protocol === protocol.value).map(r => r.model);
    model.replaceChildren(...models.map(v => new Option(v, v)));
    if (models.includes(selected)) model.value = selected;
  };
  const policyDecision = (score, risk) => {
    const p = report.enforcement_policy;
    const adjustment = risk === "write" ? p.write_adjustment : risk === "privileged" ? p.privileged_adjustment : 0;
    const block = Math.max(0, Math.min(1, p.block_threshold - adjustment));
    const review = Math.max(0, block - p.review_margin);
    return { decision: score >= block ? "BLOCK" : score >= review ? "REVIEW" : "ALLOW", block, review };
  };
  const renderPolicy = () => {
    const score = Number(document.querySelector("#score").value);
    document.querySelector("#scoreValue").value = score.toFixed(2);
    document.querySelector("#policy").innerHTML = report.enforcement_policy.tool_risks.map(risk => {
      const result = policyDecision(score, risk);
      return `<article class="policy-card ${result.decision.toLowerCase()}"><span>${risk} tool</span><strong>${result.decision}</strong><small>review ≥ ${result.review.toFixed(2)} · block ≥ ${result.block.toFixed(2)}</small></article>`;
    }).join("");
  };
  const render = () => {
    const row = report.results.find(r => r.protocol === protocol.value && r.model === model.value);
    const fields = row ? [
      ["ROC-AUC", fmt(row.roc_auc_mean)], ["PR-AUC", fmt(row.pr_auc_mean)],
      ["FPR at TPR 90", fmt(row.fpr_at_tpr_90_mean)], ["Median latency", row.latency_p50_ms_mean == null ? "Not reported" : `${row.latency_p50_ms_mean.toFixed(1)} ms`],
      ["Reported runs", String(row.n_seeds)], ["Seed status", row.seed_status]
    ] : [];
    document.querySelector("#metrics").innerHTML = fields.map(([k,v]) => `<div class="card"><span class="value">${v}</span>${k}</div>`).join("");
    const protocolRows = report.results.filter(r => r.protocol === protocol.value && r.roc_auc_mean != null).sort((a,b) => b.roc_auc_mean - a.roc_auc_mean);
    document.querySelector("#comparison").innerHTML = protocolRows.map(r => `<div class="bar-row"><code>${r.model}</code><div class="bar-track"><span style="width:${r.roc_auc_mean * 100}%"></span></div><strong>${fmt(r.roc_auc_mean)}</strong></div>`).join("");
  };
  document.querySelector("#provenance").innerHTML = [
    ["Report schema", report.schema_version], ["Experiment commit", shortHash(report.commit)],
    ["Configuration", shortHash(report.configuration_sha256)],
    ["Policy", `${report.enforcement_policy.version} · ${shortHash(report.enforcement_policy.configuration_sha256)}`],
    ["Verified split files", String(Object.keys(report.split_sha256).length)],
  ].map(([key,value]) => `<div><dt>${key}</dt><dd><code>${value}</code></dd></div>`).join("");
  document.querySelector("#limitations").innerHTML = report.limitations
    .map(limitation => `<li>${limitation}</li>`)
    .join("");
  protocol.onchange = () => { syncModels(); render(); };
  model.onchange = render;
  document.querySelector("#score").oninput = renderPolicy;
  syncModels(); render(); renderPolicy();
}).catch(error => {
  document.querySelector("#metrics").innerHTML = `<div class="card"><span class="value">Unavailable</span>${error.message}</div>`;
});
