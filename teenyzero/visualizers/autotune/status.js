function fmtNumber(value, digits = 1) {
    const num = Number(value || 0);
    return Number.isFinite(num) ? num.toFixed(digits) : "0.0";
}

function fmtInt(value) {
    const num = Number(value || 0);
    return Number.isFinite(num) ? Math.round(num).toLocaleString() : "0";
}

function fmtDate(ts) {
    if (!ts) return "n/a";
    return new Date(ts * 1000).toLocaleString();
}

function prettyStatus(value) {
    return String(value || "idle")
        .split("_")
        .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
        .join(" ");
}

function axisMarkup(width, height, padding, minValue, maxValue) {
    const usableHeight = height - padding.top - padding.bottom;
    const ticks = [0, 0.25, 0.5, 0.75, 1].map((ratio) => {
        const y = padding.top + usableHeight * ratio;
        const value = maxValue - ratio * (maxValue - minValue);
        return `
            <line x1="${padding.left}" y1="${y}" x2="${width - padding.right}" y2="${y}" stroke="#e6eaee" stroke-width="1" />
            <text x="${padding.left - 8}" y="${y + 4}" text-anchor="end" font-size="10" fill="#66727f">${value.toFixed(maxValue >= 100 ? 0 : 2)}</text>
        `;
    }).join("");
    return `
        ${ticks}
        <line x1="${padding.left}" y1="${height - padding.bottom}" x2="${width - padding.right}" y2="${height - padding.bottom}" stroke="#cfd6dd" stroke-width="1" />
    `;
}

function valueRange(values) {
    const finite = values.filter((value) => Number.isFinite(value));
    if (!finite.length) return [0, 1];
    const min = Math.min(...finite);
    const max = Math.max(...finite);
    if (min === max) return [Math.min(0, min - 1), max + 1];
    return [Math.min(min, 0), max];
}

function renderScoreChart(trials) {
    const svg = document.getElementById("score-chart");
    const width = 520;
    const height = 220;
    const padding = { top: 16, right: 12, bottom: 28, left: 46 };
    const successful = trials.filter((trial) => trial.status === "ok");
    svg.innerHTML = "";
    if (!successful.length) return;

    const values = successful.map((trial) => Number(trial.score || 0));
    const [minValue, maxValue] = valueRange(values);
    const span = Math.max(1e-6, maxValue - minValue);
    const usableWidth = width - padding.left - padding.right;
    const usableHeight = height - padding.top - padding.bottom;
    const barWidth = usableWidth / Math.max(1, successful.length);

    const bars = successful.map((trial, index) => {
        const value = Number(trial.score || 0);
        const normalized = (value - minValue) / span;
        const barHeight = normalized * usableHeight;
        const x = padding.left + index * barWidth + 6;
        const y = height - padding.bottom - barHeight;
        const fill = trial.is_baseline ? "#1a9b53" : "#2155d6";
        return `
            <rect x="${x}" y="${y}" width="${Math.max(12, barWidth - 12)}" height="${Math.max(2, barHeight)}" rx="10" fill="${fill}" />
            <text x="${x + Math.max(12, barWidth - 12) / 2}" y="${height - 8}" text-anchor="middle" font-size="10" fill="#66727f">${trial.label}</text>
        `;
    }).join("");

    svg.innerHTML = axisMarkup(width, height, padding, minValue, maxValue) + bars;
}

function renderTradeoffChart(trials) {
    const svg = document.getElementById("tradeoff-chart");
    const width = 520;
    const height = 220;
    const padding = { top: 16, right: 16, bottom: 36, left: 54 };
    const successful = trials.filter((trial) => trial.status === "ok");
    svg.innerHTML = "";
    if (!successful.length) return;

    const xValues = successful.map((trial) => Number(trial.selfplay?.positions_per_s || 0));
    const yValues = successful.map((trial) => Number(trial.train?.samples_per_s || 0));
    const [minX, maxX] = valueRange(xValues);
    const [minY, maxY] = valueRange(yValues);
    const xSpan = Math.max(1e-6, maxX - minX);
    const ySpan = Math.max(1e-6, maxY - minY);
    const usableWidth = width - padding.left - padding.right;
    const usableHeight = height - padding.top - padding.bottom;

    const points = successful.map((trial) => {
        const x = padding.left + ((Number(trial.selfplay?.positions_per_s || 0) - minX) / xSpan) * usableWidth;
        const y = padding.top + usableHeight - ((Number(trial.train?.samples_per_s || 0) - minY) / ySpan) * usableHeight;
        const fill = trial.is_baseline ? "#1a9b53" : "#111111";
        return `
            <circle cx="${x}" cy="${y}" r="6" fill="${fill}" />
            <text x="${x + 8}" y="${y - 8}" font-size="10" fill="#66727f">${trial.label}</text>
        `;
    }).join("");

    svg.innerHTML = `
        ${axisMarkup(width, height, padding, minY, maxY)}
        <text x="${width / 2}" y="${height - 8}" text-anchor="middle" font-size="10" fill="#66727f">Self-play positions/sec</text>
        <text x="18" y="${height / 2}" text-anchor="middle" font-size="10" fill="#66727f" transform="rotate(-90 18 ${height / 2})">Train samples/sec</text>
        ${points}
    `;
}

function renderSummary(latest) {
    const summary = document.getElementById("summary");
    const best = latest.best_trial || null;
    const cards = [
        ["Status", prettyStatus(latest.status), "phase 1 runner state"],
        ["Objective", latest.objective || "balanced", "how trial scores are ranked"],
        ["Profile", latest.runtime_args?.profile || "n/a", "runtime profile used during the sweep"],
        ["Trials", fmtInt(latest.trials?.length || 0), "completed phase 1 trial records"],
        ["Best Trial", best ? best.label : "n/a", "highest score in the latest ranked run"],
        ["Best Score", best ? fmtNumber(best.score, 3) : "n/a", "phase 1 score for the current best trial"],
        ["Updated", fmtDate(latest.finished_at || latest.started_at), "latest saved autotune state"],
    ];

    summary.innerHTML = cards.map(([label, value, note]) => `
        <article class="metric-card">
            <span class="metric-label">${label}</span>
            <div class="metric-value ${String(value).length > 18 ? "metric-value--compact" : ""}">${value}</div>
            <div class="metric-note">${note}</div>
        </article>
    `).join("");
}

function renderBest(latest) {
    const bestBadge = document.getElementById("best-badge");
    const bestTrial = document.getElementById("best-trial");
    const applyCommand = document.getElementById("apply-command");
    const best = latest.best_trial || null;

    bestBadge.textContent = prettyStatus(latest.status);
    if (!best) {
        bestTrial.innerHTML = `<div class="detail-row"><div class="detail-key">State</div><div class="detail-value">No phase 1 results yet</div></div>`;
        applyCommand.textContent = "Run `python3 scripts/autotune.py` to generate a phase 1 sweep.";
        return;
    }

    const config = best.config || {};
    bestTrial.innerHTML = [
        ["Trial", best.label],
        ["Score", fmtNumber(best.score, 3)],
        ["Actor Mode", config.actor_mode],
        ["Self-Play Workers", fmtInt(config.selfplay_workers)],
        ["Leaf Batch Size", fmtInt(config.selfplay_leaf_batch_size)],
        ["Train Batch Size", fmtInt(config.train_batch_size)],
        ["Train Workers", fmtInt(config.train_num_workers)],
        ["Train Precision", String(config.train_precision || "fp32").toUpperCase()],
        ["Train Compile", config.train_compile ? "yes" : "no"],
        ["Train Pin Memory", config.train_pin_memory ? "yes" : "no"],
        ["Self-Play Pos/Sec", fmtNumber(best.selfplay?.positions_per_s, 1)],
        ["Train Samples/Sec", fmtNumber(best.train?.samples_per_s, 1)],
    ].map(([key, value]) => `
        <div class="detail-row">
            <div class="detail-key">${key}</div>
            <div class="detail-value">${value}</div>
        </div>
    `).join("");

    applyCommand.textContent = latest.apply_command || "No apply command available.";
}

function renderRecent(runs) {
    const count = document.getElementById("recent-count");
    const recent = document.getElementById("recent-runs");
    count.textContent = `${runs.length} runs`;
    if (!runs.length) {
        recent.innerHTML = `<div class="detail-row"><div class="detail-key">History</div><div class="detail-value">No archived runs yet</div></div>`;
        return;
    }

    recent.innerHTML = runs.map((run) => {
        const best = run.best_trial || {};
        return `
            <div class="detail-row">
                <div>
                    <div>${run.run_id || "phase1"}</div>
                    <div class="detail-key">${prettyStatus(run.status)} · ${run.objective || "balanced"} · ${fmtInt(run.trials?.length || 0)} trials</div>
                </div>
                <div class="detail-value">${best.label ? `${best.label} (${fmtNumber(best.score, 3)})` : "n/a"}</div>
            </div>
        `;
    }).join("");
}

function renderDetails(latest) {
    const hardware = document.getElementById("hardware");
    const searchSettings = document.getElementById("search-settings");
    const hw = latest.hardware || {};
    const cuda = hw.cuda_device || {};
    const settings = latest.search_settings || {};

    hardware.innerHTML = [
        ["Device", hw.device || "n/a"],
        ["Runtime Profile", hw.runtime_profile || "n/a"],
        ["Platform", [hw.platform?.system, hw.platform?.release, hw.platform?.machine].filter(Boolean).join(" ") || "n/a"],
        ["Python", hw.platform?.python_version || "n/a"],
        ["Torch", hw.torch_version || "n/a"],
        ["CPU Count", fmtInt(hw.cpu_count || 0)],
        ["CUDA Device", cuda.name || "n/a"],
        ["VRAM", cuda.total_memory_bytes ? `${fmtNumber(cuda.total_memory_bytes / (1024 ** 3), 1)} GB` : "n/a"],
    ].map(([key, value]) => `
        <div class="detail-row">
            <div class="detail-key">${key}</div>
            <div class="detail-value">${value}</div>
        </div>
    `).join("");

    searchSettings.innerHTML = [
        ["Objective", latest.objective || "balanced"],
        ["Requested Trials", fmtInt(settings.trials || 0)],
        ["Searches / Worker", fmtInt(settings.searches_per_worker || 0)],
        ["Self-Play Sims", fmtInt(settings.selfplay_simulations || 0)],
        ["Train Batches", fmtInt(settings.train_batches || 0)],
        ["Time Budget", `${fmtNumber(settings.time_budget_minutes || 0, 1)} min`],
        ["Trial Timeout", `${fmtNumber(settings.trial_timeout_s || 0, 0)} s`],
        ["Board Backend", latest.runtime_args?.board_backend || "auto"],
    ].map(([key, value]) => `
        <div class="detail-row">
            <div class="detail-key">${key}</div>
            <div class="detail-value">${value}</div>
        </div>
    `).join("");
}

function renderTrials(latest) {
    const container = document.getElementById("trials");
    const trials = latest.trials || [];
    if (!trials.length) {
        container.innerHTML = `<div class="trial-row"><div><div class="trial-name">No phase 1 trials recorded yet</div><div class="trial-submeta">Run \`python3 scripts/autotune.py\` to populate this page.</div></div></div>`;
        return;
    }

    container.innerHTML = trials.map((trial) => {
        const config = trial.config || {};
        const bad = trial.status !== "ok";
        return `
            <div class="trial-row">
                <div>
                    <div class="trial-head">
                        <span class="trial-tag ${trial.is_baseline ? "trial-tag--baseline" : ""}">${trial.label}</span>
                        <div class="trial-name">${bad ? "Failed Trial" : "Runtime Candidate"}</div>
                    </div>
                    <div class="trial-submeta">
                        mode=${config.actor_mode} · workers=${fmtInt(config.selfplay_workers)} · leaf=${fmtInt(config.selfplay_leaf_batch_size)} · train_batch=${fmtInt(config.train_batch_size)} · train_workers=${fmtInt(config.train_num_workers)} · ${String(config.train_precision || "fp32").toUpperCase()} ${config.train_compile ? "compile" : "eager"}
                    </div>
                    ${bad && trial.errors?.length ? `<div class="trial-submeta">error: ${trial.errors.join(" | ")}</div>` : ""}
                </div>
                <div class="trial-cell">
                    <div class="trial-label">Score</div>
                    <div class="trial-number ${bad ? "trial-number--bad" : ""}">${bad ? "fail" : fmtNumber(trial.score, 3)}</div>
                </div>
                <div class="trial-cell">
                    <div class="trial-label">Pos/Sec</div>
                    <div class="trial-number">${bad ? "n/a" : fmtNumber(trial.selfplay?.positions_per_s, 1)}</div>
                </div>
                <div class="trial-cell">
                    <div class="trial-label">Train S/S</div>
                    <div class="trial-number">${bad ? "n/a" : fmtNumber(trial.train?.samples_per_s, 1)}</div>
                </div>
                <div class="trial-cell">
                    <div class="trial-label">Move ms</div>
                    <div class="trial-number">${bad ? "n/a" : fmtNumber(trial.selfplay?.move_total_mean_ms, 1)}</div>
                </div>
            </div>
        `;
    }).join("");

    renderScoreChart(trials);
    renderTradeoffChart(trials);
}

function renderRecommendations(payload) {
    const count = document.getElementById("recommendation-count");
    const container = document.getElementById("recommendations");
    const items = payload?.recommendations || [];
    count.textContent = `${items.length} promoted`;
    if (!items.length) {
        container.innerHTML = `<div class="trial-row"><div><div class="trial-name">No shared recommendations yet</div><div class="trial-submeta">Promote a run with \`python3 scripts/promote_autotune.py\` and it will show up here.</div></div></div>`;
        return;
    }

    container.innerHTML = items.map((entry) => `
        <div class="trial-row">
            <div>
                <div class="trial-head">
                    <span class="trial-tag">${entry.workload || "run"}</span>
                    <div class="trial-name">${entry.title || entry.id}</div>
                </div>
                <div class="trial-submeta">
                    ${entry.device_family || "unknown"} · seed=${entry.runtime_seed?.device || "n/a"}/${entry.runtime_seed?.profile || "n/a"}/${entry.runtime_seed?.board_backend || "n/a"} · trial=${entry.source?.best_trial_label || "n/a"}
                </div>
                <div class="trial-submeta">${entry.summary || ""}</div>
                <div class="trial-submeta mono">${entry.apply_command || ""}</div>
            </div>
            <div class="trial-cell">
                <div class="trial-label">Phase 1</div>
                <div class="trial-number">${fmtNumber(entry.metrics?.phase1_score, 3)}</div>
            </div>
            <div class="trial-cell">
                <div class="trial-label">Pos/Sec</div>
                <div class="trial-number">${fmtNumber(entry.metrics?.selfplay_positions_per_s, 1)}</div>
            </div>
            <div class="trial-cell">
                <div class="trial-label">Train S/S</div>
                <div class="trial-number">${fmtNumber(entry.metrics?.train_samples_per_s, 1)}</div>
            </div>
            <div class="trial-cell">
                <div class="trial-label">Move ms</div>
                <div class="trial-number">${fmtNumber(entry.metrics?.selfplay_move_total_mean_ms, 1)}</div>
            </div>
        </div>
    `).join("");
}

async function load() {
    const [latestResponse, runsResponse, recsResponse] = await Promise.all([
        fetch("/api/autotune_status"),
        fetch("/api/autotune_runs"),
        fetch("/api/autotune_recommendations"),
    ]);
    const latest = await latestResponse.json();
    const runs = await runsResponse.json();
    const recommendations = await recsResponse.json();
    renderSummary(latest);
    renderBest(latest);
    renderRecent(runs || []);
    renderDetails(latest);
    renderRecommendations(recommendations);
    renderTrials(latest);
}

load().catch((error) => {
    console.error(error);
});

setInterval(() => {
    load().catch((error) => console.error(error));
}, 5000);
