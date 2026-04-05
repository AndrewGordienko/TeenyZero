function prettyStatus(value) {
    return String(value || "idle")
        .split("_")
        .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
        .join(" ");
}

const pieceMap = {
    P: "♙",
    N: "♘",
    B: "♗",
    R: "♖",
    Q: "♕",
    K: "♔",
    p: "♟",
    n: "♞",
    b: "♝",
    r: "♜",
    q: "♛",
    k: "♚",
};
let sampleRefreshNonce = Date.now();

function setHtml(id, html) {
    const node = document.getElementById(id);
    if (node) {
        node.innerHTML = html;
    }
}

function fmtNumber(value, digits = 1) {
    const num = Number(value || 0);
    return Number.isFinite(num) ? num.toFixed(digits) : "0.0";
}

function fmtLoss(value) {
    const num = Number(value || 0);
    if (!Number.isFinite(num)) return "0.0000";
    if (num === 0) return "0.0000";
    if (Math.abs(num) < 0.001) return num.toExponential(2);
    return num.toFixed(4);
}

function fmtInt(value) {
    const num = Number(value || 0);
    return Number.isFinite(num) ? Math.round(num).toLocaleString() : "0";
}

function fmtDate(ts) {
    if (!ts) return "n/a";
    return new Date(ts * 1000).toLocaleString();
}

function fmtDuration(seconds) {
    const value = Number(seconds || 0);
    if (!Number.isFinite(value)) return "0.0s";
    if (value >= 60) return `${(value / 60).toFixed(1)}m`;
    if (value >= 1) return `${value.toFixed(1)}s`;
    return `${(value * 1000).toFixed(0)}ms`;
}

function stageElapsed(state) {
    const started = Number(state.started_at || 0);
    const heartbeat = Number(state.heartbeat_at || 0);
    if (!started || !heartbeat) return 0;
    return Math.max(0, heartbeat - started);
}

function buildLiveHistory(state, history) {
    const next = history.slice();
    if (!String(state.status || "").includes("training")) {
        return next;
    }
    next.push({
        epoch: Number(state.last_epoch || 0),
        finished_at: Number(state.heartbeat_at || 0),
        loss: Number(state.running_loss || state.last_loss || 0),
        attack_loss: Number(state.running_attack_loss || state.last_attack_loss || 0),
        pressure_loss: Number(state.running_pressure_loss || state.last_pressure_loss || 0),
        samples_per_s: Number(state.last_samples_per_s || 0),
        batches_per_s: Number(state.last_batches_per_s || 0),
        avg_batch_time_ms: Number(state.avg_batch_time_ms || 0),
        duration_s: Number(state.elapsed_s || 0),
        is_live: true,
    });
    return next;
}

function render(state) {
    const summary = document.getElementById("summary");
    const progress = document.getElementById("progress");
    const details = document.getElementById("details");
    const stageCards = document.getElementById("stage-cards");
    const throughputCards = document.getElementById("throughput-cards");
    const profile = state.runtime_profile_settings || {};

    const epochProgress = Number(state.last_epoch || 0);
    const totalEpochs = Number(state.epochs || 0);
    const batchProgress = Number(state.completed_batches || 0);
    const batchTotal = Number(state.total_batches || 0);
    const positionsPerEpoch = Number(state.positions_per_epoch || 0);
    const minPlies = Number(state.min_plies || 0);
    const maxPlies = Number(state.max_plies || 0);

    const cards = [
        ["Status", prettyStatus(state.status), "current pretraining stage"],
        ["Profile", state.runtime_profile || "local", "active runtime profile"],
        ["Heartbeat", fmtDate(state.heartbeat_at), "latest pretrainer update"],
        ["Last Epoch", fmtInt(state.last_epoch), "most recently completed or active epoch"],
        ["Total Loss", fmtLoss(state.running_loss || state.last_loss), "combined geometry loss"],
        ["Checkpoint", state.checkpoint_path || "n/a", "latest AlphaFold pretraining checkpoint"],
    ];

    summary.innerHTML = cards.map(([label, value, note]) => `
        <article class="metric-card">
            <span class="metric-label">${label}</span>
            <div class="metric-value ${String(value).length > 18 ? "metric-value--compact" : ""}">${value}</div>
            <div class="metric-note">${note}</div>
        </article>
    `).join("");

    const progressCards = [
        [
            "Epoch Progress",
            `${fmtInt(epochProgress)} / ${fmtInt(totalEpochs)}`,
            Math.max(0, Math.min(100, totalEpochs ? (epochProgress / totalEpochs) * 100 : 0)),
            `${fmtInt(Math.max(0, totalEpochs - epochProgress))} epochs remaining`,
        ],
        [
            "Batch Progress",
            `${fmtInt(batchProgress)} / ${fmtInt(batchTotal)}`,
            Math.max(0, Math.min(100, batchTotal ? (batchProgress / batchTotal) * 100 : 0)),
            "batches completed in the active epoch",
        ],
        [
            "Position Coverage",
            fmtInt(positionsPerEpoch),
            100,
            `${fmtInt(minPlies)}-${fmtInt(maxPlies)} random plies before each sampled board`,
        ],
    ];

    progress.innerHTML = progressCards.map(([title, value, pct, note]) => `
        <section class="panel progress-card">
            <h2>${title}</h2>
            <div class="progress-value">${value}</div>
            <div class="progress-bar"><div class="progress-fill" style="width:${pct.toFixed(1)}%"></div></div>
            <div class="progress-meta">
                <span>${note}</span>
                <span>${pct.toFixed(0)}%</span>
            </div>
        </section>
    `).join("");

    const stageInfo = [
        ["Run Status", prettyStatus(state.status), `${fmtDuration(stageElapsed(state))} since the current run began`],
        ["Geometry Loss", fmtLoss(state.running_loss || state.last_loss), `${fmtLoss(state.running_attack_loss || state.last_attack_loss)} attack and ${fmtLoss(state.running_pressure_loss || state.last_pressure_loss)} pressure`],
        ["Position Sampler", `${fmtInt(minPlies)}-${fmtInt(maxPlies)} plies`, `${fmtInt(positionsPerEpoch)} boards per epoch with seed ${fmtInt(state.seed)}`],
    ];

    stageCards.innerHTML = stageInfo.map(([label, value, note]) => `
        <div class="info-card">
            <div class="info-label">${label}</div>
            <div class="info-value">${value}</div>
            <div class="info-note">${note}</div>
        </div>
    `).join("");

    const throughputInfo = [
        ["Samples / Sec", fmtInt(state.last_samples_per_s), "position throughput in the latest or active epoch"],
        ["Batches / Sec", fmtNumber(state.last_batches_per_s, 2), "optimizer batch rate through the geometry objective"],
        ["Batch Time", `${fmtNumber(state.avg_batch_time_ms, 1)} ms`, `${fmtInt(state.batch_size || profile.train_batch_size || 0)} samples per batch`],
    ];

    throughputCards.innerHTML = throughputInfo.map(([label, value, note]) => `
        <div class="info-card">
            <div class="info-label">${label}</div>
            <div class="info-value">${value}</div>
            <div class="info-note">${note}</div>
        </div>
    `).join("");

    const sections = [
        [
            "Run Configuration",
            [
                ["Device", state.device || "n/a"],
                ["Precision", String(state.precision || profile.train_precision || "fp32").toUpperCase()],
                ["Learning Rate", fmtNumber(state.lr, 6)],
                ["Weight Decay", fmtNumber(state.weight_decay, 6)],
                ["Batch Size", fmtInt(state.batch_size)],
                ["Loader Workers", fmtInt(state.num_workers)],
                ["Checkpoint Path", state.checkpoint_path || "n/a"],
            ],
        ],
        [
            "Geometry Targets",
            [
                ["Objective", "Attack maps + king pressure"],
                ["Attack Loss", fmtLoss(state.running_attack_loss || state.last_attack_loss)],
                ["Pressure Loss", fmtLoss(state.running_pressure_loss || state.last_pressure_loss)],
                ["Positions / Epoch", fmtInt(state.positions_per_epoch)],
                ["Min Plies", fmtInt(state.min_plies)],
                ["Max Plies", fmtInt(state.max_plies)],
            ],
        ],
        [
            "Lifecycle",
            [
                ["Started", fmtDate(state.started_at)],
                ["Finished", fmtDate(state.finished_at)],
                ["Heartbeat", fmtDate(state.heartbeat_at)],
                ["Epoch", `${fmtInt(state.last_epoch)} / ${fmtInt(state.epochs)}`],
                ["Completed Batches", `${fmtInt(state.completed_batches)} / ${fmtInt(state.total_batches)}`],
                ["Runtime Profile", state.runtime_profile || "n/a"],
            ],
        ],
    ];

    details.innerHTML = sections.map(([title, rows]) => `
        <section class="detail-section">
            <h3 class="detail-section-title">${title}</h3>
            <div class="detail-list">
                ${rows.map(([key, value]) => `
                    <div class="detail-row">
                        <div class="detail-key">${key}</div>
                        <div class="detail-value">${value}</div>
                    </div>
                `).join("")}
            </div>
        </section>
    `).join("");
}

function renderLoadError(message) {
    setHtml("summary", `
        <article class="metric-card">
            <span class="metric-label">Dashboard Error</span>
            <div class="metric-value metric-value--compact">AlphaFold data unavailable</div>
            <div class="metric-note">${message}</div>
        </article>
    `);
    setHtml("progress", "");
    setHtml("details", "");
    setHtml("stage-cards", "");
    setHtml("throughput-cards", "");
    setHtml("sample-meta", "");
    setHtml("sample-board", "");
    setHtml("target-maps", "");
    setHtml("prediction-maps", "");
    setHtml("connections", "");
    for (const id of ["loss-chart", "throughput-chart", "duration-chart", "batch-chart"]) {
        const svg = document.getElementById(id);
        if (svg) svg.innerHTML = "";
    }
}

function seriesRange(series) {
    const values = series.flatMap((item) => item.values);
    if (!values.length) return [0, 1];
    const minValue = Math.min(...values);
    const maxValue = Math.max(...values);
    if (minValue === maxValue) return [Math.min(0, minValue), maxValue + 1];
    return [Math.min(0, minValue), maxValue];
}

function axisMarkup(width, height, padding, minValue, maxValue, xLabels) {
    const usableHeight = height - padding.top - padding.bottom;
    const yTicks = [0, 0.25, 0.5, 0.75, 1].map((ratio) => {
        const y = padding.top + usableHeight * ratio;
        const value = maxValue - ratio * (maxValue - minValue);
        return `
            <line x1="${padding.left}" y1="${y}" x2="${width - padding.right}" y2="${y}" stroke="#e6eaee" stroke-width="1" />
            <text x="${padding.left - 8}" y="${y + 4}" text-anchor="end" font-size="10" fill="#66727f">${value.toFixed(maxValue >= 10 ? 0 : 2)}</text>
        `;
    }).join("");
    return `
        ${yTicks}
        <line x1="${padding.left}" y1="${height - padding.bottom}" x2="${width - padding.right}" y2="${height - padding.bottom}" stroke="#cfd6dd" stroke-width="1" />
        <text x="${padding.left}" y="${height - 6}" font-size="10" fill="#66727f">${xLabels[0]}</text>
        <text x="${width - padding.right}" y="${height - 6}" text-anchor="end" font-size="10" fill="#66727f">${xLabels[1]}</text>
    `;
}

function ensureTooltip(svgId) {
    const svg = document.getElementById(svgId);
    let tooltip = svg.parentElement.querySelector(".chart-tooltip");
    if (!tooltip) {
        tooltip = document.createElement("div");
        tooltip.className = "chart-tooltip";
        svg.parentElement.appendChild(tooltip);
    }
    return tooltip;
}

function attachTooltip(svgId, pointCount, formatter) {
    const svg = document.getElementById(svgId);
    const tooltip = ensureTooltip(svgId);
    if (pointCount <= 0) {
        tooltip.style.opacity = "0";
        return;
    }
    const width = 520;
    const padding = { left: 44, right: 14 };
    const usableWidth = width - padding.left - padding.right;
    svg.onmousemove = (event) => {
        const rect = svg.getBoundingClientRect();
        const localX = ((event.clientX - rect.left) / rect.width) * width;
        const clamped = Math.max(padding.left, Math.min(width - padding.right, localX));
        const ratio = (clamped - padding.left) / Math.max(1, usableWidth);
        const index = Math.max(0, Math.min(pointCount - 1, Math.round(ratio * Math.max(1, pointCount - 1))));
        tooltip.innerHTML = formatter(index);
        tooltip.style.opacity = "1";
        const scaledX = (clamped / width) * rect.width;
        tooltip.style.left = `${Math.min(rect.width - 196, Math.max(12, scaledX + 12))}px`;
    };
    svg.onmouseleave = () => {
        tooltip.style.opacity = "0";
    };
}

function renderMultiLineChart(svgId, series, xLabels) {
    const svg = document.getElementById(svgId);
    const width = 520;
    const height = 180;
    const padding = { top: 14, right: 14, bottom: 24, left: 44 };
    svg.innerHTML = "";
    if (!series.length || !series.some((item) => item.values.length)) return;

    const [minValue, maxValue] = seriesRange(series);
    const valueSpan = Math.max(1e-6, maxValue - minValue);
    const usableWidth = width - padding.left - padding.right;
    const usableHeight = height - padding.top - padding.bottom;
    const lines = series.map((item) => {
        if (!item.values.length) return "";
        const points = item.values.map((value, index) => {
            const x = padding.left + (index * usableWidth) / Math.max(1, item.values.length - 1);
            const y = padding.top + usableHeight - ((value - minValue) / valueSpan) * usableHeight;
            return `${x},${y}`;
        }).join(" ");
        return `<polyline fill="none" stroke="${item.color}" stroke-width="3" points="${points}" />`;
    }).join("");
    svg.innerHTML = axisMarkup(width, height, padding, minValue, maxValue, xLabels) + lines;
}

function historyLabel(item, index) {
    if (item.is_live) return "Live";
    const epoch = Number(item.epoch || index + 1);
    return `Epoch ${epoch}`;
}

function renderHistory(state, history) {
    const recent = buildLiveHistory(state, history).slice(-24);
    const xLabels = recent.length ? [historyLabel(recent[0], 0), historyLabel(recent[recent.length - 1], recent.length - 1)] : ["Oldest", "Latest"];

    renderMultiLineChart("loss-chart", [
        { values: recent.map((item) => Number(item.loss || 0)), color: "#2155d6" },
        { values: recent.map((item) => Number(item.attack_loss || 0)), color: "#111111" },
        { values: recent.map((item) => Number(item.pressure_loss || 0)), color: "#1a9b53" },
    ], xLabels);
    renderMultiLineChart("throughput-chart", [
        { values: recent.map((item) => Number(item.samples_per_s || 0)), color: "#2155d6" },
        { values: recent.map((item) => Number(item.batches_per_s || 0)), color: "#c58f12" },
    ], xLabels);
    renderMultiLineChart("duration-chart", [
        { values: recent.map((item) => Number(item.duration_s || 0)), color: "#b04f73" },
    ], xLabels);
    renderMultiLineChart("batch-chart", [
        { values: recent.map((item) => Number(item.avg_batch_time_ms || 0)), color: "#111111" },
    ], xLabels);

    attachTooltip("loss-chart", recent.length, (index) => `
        <strong>${historyLabel(recent[index], index)}</strong>
        Total Loss: ${fmtLoss(recent[index].loss)}<br>
        Attack Loss: ${fmtLoss(recent[index].attack_loss)}<br>
        Pressure Loss: ${fmtLoss(recent[index].pressure_loss)}
    `);
    attachTooltip("throughput-chart", recent.length, (index) => `
        <strong>${historyLabel(recent[index], index)}</strong>
        Samples / Sec: ${fmtInt(recent[index].samples_per_s)}<br>
        Batches / Sec: ${fmtNumber(recent[index].batches_per_s, 2)}
    `);
    attachTooltip("duration-chart", recent.length, (index) => `
        <strong>${historyLabel(recent[index], index)}</strong>
        Duration: ${fmtDuration(recent[index].duration_s)}<br>
        Samples Seen: ${fmtInt(recent[index].samples_seen)}
    `);
    attachTooltip("batch-chart", recent.length, (index) => `
        <strong>${historyLabel(recent[index], index)}</strong>
        Avg Batch Time: ${fmtNumber(recent[index].avg_batch_time_ms, 1)} ms<br>
        Batches: ${fmtInt(recent[index].batches)}
    `);
}

function renderBoard(fen) {
    const boardPart = (fen || "").split(" ")[0] || "8/8/8/8/8/8/8/8";
    const rows = boardPart.split("/");
    const squares = [];
    rows.forEach((row, rankIndex) => {
        let fileIndex = 0;
        for (const char of row) {
            if (Number.isInteger(Number(char)) && char !== "0") {
                for (let i = 0; i < Number(char); i += 1) {
                    const isLight = (rankIndex + fileIndex) % 2 === 0;
                    squares.push(`<div class="sample-square ${isLight ? "light" : "dark"}"></div>`);
                    fileIndex += 1;
                }
            } else {
                const isLight = (rankIndex + fileIndex) % 2 === 0;
                const pieceClass = char === char.toUpperCase() ? "white" : "black";
                squares.push(
                    `<div class="sample-square ${isLight ? "light" : "dark"}"><span class="piece ${pieceClass}">${pieceMap[char] || ""}</span></div>`,
                );
                fileIndex += 1;
            }
        }
    });
    return squares.join("");
}

function heatColor(value, hue = 220) {
    const v = Math.max(0, Math.min(1, Number(value || 0)));
    const alpha = 0.08 + v * 0.84;
    return `hsla(${hue}, 68%, 48%, ${alpha})`;
}

function renderHeatmapMaps(target, maps, hue) {
    if (!maps || typeof maps !== "object") {
        target.innerHTML = '<div class="info-note">No map data available.</div>';
        return;
    }
    target.innerHTML = Object.entries(maps).map(([name, grid]) => `
        <section class="heatmap-card">
            <div class="heatmap-title">${name.replaceAll("_", " ")}</div>
            <div class="heatmap-grid">
                ${grid.flatMap((row) => row.map((value) => `
                    <div class="heatmap-cell" style="background:${heatColor(value, hue)}" title="${fmtNumber(value, 3)}"></div>
                `)).join("")}
            </div>
        </section>
    `).join("");
}

function renderConnections(target, connections) {
    if (!Array.isArray(connections) || !connections.length) {
        target.innerHTML = '<div class="info-note">No learned connection data is available yet.</div>';
        return;
    }
    target.innerHTML = connections.map((item) => `
        <div class="connection-row">
            <div class="connection-label mono">${item.from} → ${item.to}</div>
            <div class="connection-weight">${fmtNumber(item.weight, 3)}</div>
        </div>
    `).join("");
}

function renderSample(sample) {
    if (sample && sample.error) {
        setHtml("sample-meta", `
            <div class="info-note">
                Sample inspector unavailable: ${sample.error}
            </div>
        `);
        setHtml("sample-board", "");
        setHtml("target-maps", "");
        setHtml("prediction-maps", "");
        setHtml("connections", "");
        return;
    }

    if (!sample || !sample.board_fen) {
        setHtml("sample-meta", '<div class="info-note">No geometry sample is available.</div>');
        setHtml("sample-board", "");
        setHtml("target-maps", "");
        setHtml("prediction-maps", "");
        setHtml("connections", "");
        return;
    }

    setHtml("sample-meta", [
        ["Source", sample.source || "n/a"],
        ["Checkpoint", sample.checkpoint_path || "targets only"],
        ["Board", sample.board_fen],
    ].map(([label, value]) => `
        <div class="sample-chip">
            <div class="sample-chip-label">${label}</div>
            <div class="sample-chip-value mono">${value}</div>
        </div>
    `).join(""));
    setHtml("sample-board", renderBoard(sample.board_fen));
    renderHeatmapMaps(document.getElementById("target-maps"), sample.targets, 220);
    renderHeatmapMaps(document.getElementById("prediction-maps"), sample.predictions, 145);
    renderConnections(document.getElementById("connections"), sample.top_connections);
}

async function fetchJson(url) {
    const response = await fetch(url);
    const contentType = String(response.headers.get("content-type") || "").toLowerCase();
    const text = await response.text();

    if (contentType.includes("application/json")) {
        let payload;
        try {
            payload = JSON.parse(text);
        } catch (error) {
            throw new Error(`${url} returned invalid JSON: ${error.message}`);
        }
        if (!response.ok) {
            throw new Error(payload.error || `${url} failed with status ${response.status}`);
        }
        return payload;
    }

    const compactText = text.replace(/\s+/g, " ").trim().slice(0, 160);
    if (!response.ok) {
        throw new Error(`${url} failed with status ${response.status}: ${compactText || "non-JSON response"}`);
    }
    throw new Error(`${url} returned non-JSON response: ${compactText || "empty response"}`);
}

async function update() {
    try {
        const source = document.getElementById("sample-source")?.value || "replay";
        const [state, history] = await Promise.all([
            fetchJson("/api/alphafold_status"),
            fetchJson("/api/alphafold_history"),
        ]);
        let sample = null;
        try {
            sample = await fetchJson(`/api/alphafold_sample?source=${encodeURIComponent(source)}&seed=${sampleRefreshNonce}`);
        } catch (error) {
            console.warn("AlphaFold sample inspector unavailable:", error);
            sample = { error: error.message };
        }
        render(state);
        renderHistory(state, history);
        renderSample(sample);
    } catch (error) {
        console.error("Failed to update AlphaFold dashboard:", error);
        renderLoadError(error.message);
    }
}

document.getElementById("refresh-sample")?.addEventListener("click", () => {
    sampleRefreshNonce = Date.now();
    update();
});
document.getElementById("sample-source")?.addEventListener("change", () => {
    sampleRefreshNonce = Date.now();
    update();
});
update();
setInterval(update, 5000);
