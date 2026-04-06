function prettyStatus(value) {
    return String(value || "idle")
        .split("_")
        .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
        .join(" ");
}

const HEATMAP_INFO = {
    white_attack: {
        title: "White Control / Attack",
        note: "Squares white pieces attack or defend. This is board control, not move preference.",
    },
    black_attack: {
        title: "Black Control / Attack",
        note: "Squares black pieces attack or defend. Friendly occupied squares can still light up because they are defended.",
    },
    white_king_pressure: {
        title: "White King Pressure",
        note: "White pressure on the black king zone.",
    },
    black_king_pressure: {
        title: "Black King Pressure",
        note: "Black pressure on the white king zone.",
    },
};
let sampleRefreshNonce = Date.now();
let activeConnectionSquareIndex = null;

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

function fmtMaybeNumber(value, digits = 3) {
    const num = Number(value);
    return Number.isFinite(num) ? num.toFixed(digits) : "n/a";
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

function squareName(squareIndex) {
    const index = Math.max(0, Math.min(63, Number(squareIndex || 0)));
    const file = "abcdefgh"[index % 8] || "a";
    const rank = Math.floor(index / 8) + 1;
    return `${file}${rank}`;
}

function pieceAsset(symbol) {
    if (!symbol) return "";
    const color = symbol === symbol.toUpperCase() ? "w" : "b";
    return `/static/assets/pieces/${color}${symbol.toLowerCase()}.png`;
}

function pieceName(symbol) {
    if (!symbol) return "Empty square";
    const color = symbol === symbol.toUpperCase() ? "White" : "Black";
    const nameMap = {
        p: "Pawn",
        n: "Knight",
        b: "Bishop",
        r: "Rook",
        q: "Queen",
        k: "King",
    };
    return `${color} ${nameMap[symbol.toLowerCase()] || "Piece"}`;
}

function mapInfoForSample(sample) {
    const info = sample?.map_info;
    if (info && typeof info === "object" && Object.keys(info).length) {
        return info;
    }
    return HEATMAP_INFO;
}

function mapKeysForSample(sample) {
    return Object.keys(mapInfoForSample(sample));
}

function squareMetrics(sample, squareIndex) {
    const index = Math.max(0, Math.min(63, Number(squareIndex || 0)));
    const row = Math.floor(index / 8);
    const col = index % 8;
    const pieceGrid = Array.isArray(sample?.piece_grid) ? sample.piece_grid : [];
    const piece = Array.isArray(pieceGrid[row]) ? String(pieceGrid[row][col] || "") : "";
    const mapKeys = mapKeysForSample(sample);
    return {
        square: squareName(index),
        piece,
        targetValues: Object.fromEntries(mapKeys.map((name) => [
            name,
            Number(sample?.targets?.[name]?.[row]?.[col] || 0),
        ])),
        predictionValues: Object.fromEntries(mapKeys.map((name) => [
            name,
            sample?.predictions?.[name]?.[row]?.[col] ?? Number.NaN,
        ])),
    };
}

function gridMeanAbsError(targetGrid, predictionGrid) {
    let total = 0;
    let count = 0;
    for (let row = 0; row < 8; row += 1) {
        for (let col = 0; col < 8; col += 1) {
            const target = Number(targetGrid?.[row]?.[col] || 0);
            const prediction = Number(predictionGrid?.[row]?.[col] || 0);
            total += Math.abs(target - prediction);
            count += 1;
        }
    }
    return count > 0 ? total / count : 0;
}

function strongestAttentionSource(attentionMatrix) {
    if (!Array.isArray(attentionMatrix) || attentionMatrix.length !== 64) return null;
    let bestSource = null;
    let bestWeight = -1;
    for (let source = 0; source < 64; source += 1) {
        const row = Array.isArray(attentionMatrix[source]) ? attentionMatrix[source] : [];
        for (let target = 0; target < 64; target += 1) {
            if (source === target) continue;
            const weight = Number(row[target] || 0);
            if (weight > bestWeight) {
                bestWeight = weight;
                bestSource = source;
            }
        }
    }
    return bestSource;
}

function activeSquareForSample(sample) {
    if (Number.isInteger(activeConnectionSquareIndex)) return activeConnectionSquareIndex;
    return strongestAttentionSource(sample?.attention_matrix);
}

function topConnectionsForSquare(sample, squareIndex, limit = 6) {
    const matrix = sample?.attention_matrix;
    if (!Array.isArray(matrix) || !Array.isArray(matrix[squareIndex])) return [];
    const ranked = [];
    for (let target = 0; target < 64; target += 1) {
        if (target === squareIndex) continue;
        ranked.push({
            squareIndex: target,
            square: squareName(target),
            weight: Number(matrix[squareIndex][target] || 0),
        });
    }
    ranked.sort((a, b) => b.weight - a.weight);
    return ranked.slice(0, limit);
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
        ["Position Sampler", `${fmtInt(minPlies)}-${fmtInt(maxPlies)} plies`, state.sampling_note || `${fmtInt(positionsPerEpoch)} boards per epoch with seed ${fmtInt(state.seed)}`],
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
                ["Data Source", prettyStatus(state.data_source || "weighted_random_legal_rollouts")],
                ["Objective", "Attack maps + king pressure"],
                ["Target Style", prettyStatus(state.target_style || "normalized_attack_intensity_v2")],
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
                ["Chunk Cache", fmtInt(state.cache_chunk_size || 0)],
                ["Prefetch", fmtInt(state.prefetch_factor || 0)],
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
    setHtml("sample-summary", "");
    setHtml("sample-guide", "");
    setHtml("sample-board-label", "White at bottom");
    setHtml("sample-board", "");
    hideSamplePopover();
    setHtml("geometry-compare", "");
    setHtml("connection-lens", "");
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

function renderBoard(pieceGrid, focusSquareIndex = null) {
    const grid = Array.isArray(pieceGrid) ? pieceGrid : [];
    const squares = [];

    for (let displayRow = 0; displayRow < 8; displayRow += 1) {
        const row = 7 - displayRow;
        const rowData = Array.isArray(grid[row]) ? grid[row] : [];
        for (let col = 0; col < 8; col += 1) {
            const squareIndex = row * 8 + col;
            const square = squareName(squareIndex);
            const piece = String(rowData[col] || "");
            const isLight = (displayRow + col) % 2 === 0;
            squares.push(`
                <div class="sample-square ${isLight ? "light" : "dark"} ${squareIndex === focusSquareIndex ? "is-focus" : ""}" data-square-index="${squareIndex}" title="${square}">
                    ${piece ? `<img class="piece-sprite" src="${pieceAsset(piece)}" alt="${pieceName(piece)}">` : ""}
                </div>
            `);
        }
    }

    return squares.join("");
}

function heatColor(value, hue = 220) {
    const v = Math.max(0, Math.min(1, Number(value || 0)));
    const alpha = 0.08 + v * 0.84;
    return `hsla(${hue}, 68%, 48%, ${alpha})`;
}

function mapHue(name) {
    return String(name).includes("pressure") ? 24 : 216;
}

function renderHeatGrid(grid, name, kind) {
    const cells = [];
    for (let displayRow = 0; displayRow < 8; displayRow += 1) {
        const row = 7 - displayRow;
        const rowData = Array.isArray(grid?.[row]) ? grid[row] : [];
        for (let col = 0; col < 8; col += 1) {
            const squareIndex = row * 8 + col;
            const value = Number(rowData[col] || 0);
            cells.push(`
                <div
                    class="heatmap-cell"
                    data-square-index="${squareIndex}"
                    data-map-name="${name}"
                    data-map-kind="${kind}"
                    style="background:${heatColor(value, mapHue(name))}"
                    title="${squareName(squareIndex)}: ${fmtNumber(value, 3)}"
                ></div>
            `);
        }
    }
    return `<div class="heatmap-grid">${cells.join("")}</div>`;
}

function renderSampleSummary(sample) {
    const chips = [
        ["Sample", sample.source === "random" ? "Random training position" : "Replay fine-tune position"],
        ["View", sample.board_view_label || "Board view"],
        ["Inspector", sample.predictions ? "Target + prediction loaded" : "Targets only"],
    ];
    if (sample.warning) {
        chips.push(["Warning", sample.warning]);
    }
    return chips.map(([label, value]) => `
        <div class="sample-chip">
            <div class="sample-chip-label">${label}</div>
            <div class="sample-chip-value">${value}</div>
        </div>
    `).join("");
}

function renderSampleGuide(sample) {
    return `
        <div class="sample-guide-label">What You're Looking At</div>
        <div class="sample-guide-grid">
            <div class="sample-guide-card">
                <div class="sample-guide-title">Attack / Control</div>
                <div class="sample-guide-note">Attack here means squares a side currently attacks or defends with its existing pieces. It is not where that side wants to move next. A side can "attack itself" in the sense that its own pieces are defended.</div>
            </div>
            <div class="sample-guide-card">
                <div class="sample-guide-title">Target Vs Prediction</div>
                <div class="sample-guide-note">Target geometry is computed directly from chess rules on the sampled board. Predicted geometry is the model's estimate of those same maps before it sees the label.</div>
            </div>
            <div class="sample-guide-card">
                <div class="sample-guide-title">Connections</div>
                <div class="sample-guide-note">Connections are internal square-to-square attention weights. They are not legal moves or policy scores. They show which other squares the model consults when representing one focused square.</div>
            </div>
        </div>
        ${sample?.board_view_note ? `<div class="sample-guide-note" style="margin-top:12px;">${sample.board_view_note}</div>` : ""}
    `;
}

function renderGeometryCompare(sample) {
    const target = document.getElementById("geometry-compare");
    if (!target) return;
    if (!sample?.targets) {
        target.innerHTML = '<div class="info-note">No geometry maps are available.</div>';
        return;
    }
    const mapInfo = mapInfoForSample(sample);
    const mapKeys = mapKeysForSample(sample);
    target.innerHTML = mapKeys.map((name) => {
        const meta = mapInfo[name] || { title: name, note: "" };
        const targetGrid = sample.targets?.[name];
        const predictionGrid = sample.predictions?.[name];
        const mae = predictionGrid ? gridMeanAbsError(targetGrid, predictionGrid) : null;
        return `
            <section class="compare-card">
                <div class="compare-title">${meta.title}</div>
                <div class="compare-note">${meta.note}</div>
                <div class="compare-meta">
                    <span>${predictionGrid ? `Mean abs error: ${fmtNumber(mae, 3)}` : "Prediction unavailable"}</span>
                </div>
                <div class="compare-panels">
                    <div>
                        <div class="compare-panel-label">Target</div>
                        ${renderHeatGrid(targetGrid, name, "target")}
                    </div>
                    <div>
                        <div class="compare-panel-label">Prediction</div>
                        ${predictionGrid ? renderHeatGrid(predictionGrid, name, "prediction") : '<div class="info-note">Prediction unavailable for this sample.</div>'}
                    </div>
                </div>
            </section>
        `;
    }).join("");
}

function renderConnectionBoard(sample, sourceSquareIndex) {
    const matrix = sample?.attention_matrix;
    const pieceGrid = Array.isArray(sample?.piece_grid) ? sample.piece_grid : [];
    const weights = Array.isArray(matrix?.[sourceSquareIndex]) ? matrix[sourceSquareIndex] : [];
    const cells = [];

    for (let displayRow = 0; displayRow < 8; displayRow += 1) {
        const row = 7 - displayRow;
        const rowData = Array.isArray(pieceGrid[row]) ? pieceGrid[row] : [];
        for (let col = 0; col < 8; col += 1) {
            const squareIndex = row * 8 + col;
            const piece = String(rowData[col] || "");
            const isLight = (displayRow + col) % 2 === 0;
            const weight = Number(weights[squareIndex] || 0);
            cells.push(`
                <div
                    class="connection-board-square ${isLight ? "light" : "dark"} ${squareIndex === sourceSquareIndex ? "is-focus" : ""}"
                    data-square-index="${squareIndex}"
                    style="--connection-color:${heatColor(weight, 336)}"
                    title="${squareName(squareIndex)}: ${fmtNumber(weight, 3)}"
                >
                    ${piece ? `<img class="piece-sprite" src="${pieceAsset(piece)}" alt="${pieceName(piece)}">` : ""}
                </div>
            `);
        }
    }

    return `<div class="connection-board">${cells.join("")}</div>`;
}

function renderConnectionLens(sample, sourceSquareIndex = null) {
    const node = document.getElementById("connection-lens");
    if (!node) return;
    if (!sample?.attention_matrix) {
        node.innerHTML = '<div class="info-note">Connection weights are only available when the inspector model is loaded successfully.</div>';
        return;
    }
    const focusSquare = Number.isInteger(sourceSquareIndex) ? sourceSquareIndex : activeSquareForSample(sample);
    if (!Number.isInteger(focusSquare)) {
        node.innerHTML = '<div class="info-note">Hover a square to inspect its square-to-square attention pattern.</div>';
        return;
    }
    const related = topConnectionsForSquare(sample, focusSquare, 6);
    node.innerHTML = `
        <div class="connection-meta">
            <div class="connection-headline">Focused square: <span class="mono">${squareName(focusSquare)}</span></div>
            <div class="connection-note">Colored squares show where the model places the most attention when building the representation for <span class="mono">${squareName(focusSquare)}</span>. Higher intensity means a stronger internal relation, not a stronger move.</div>
        </div>
        ${renderConnectionBoard(sample, focusSquare)}
        <div class="connection-list">
            ${related.map((item, index) => `
                <div class="connection-item">
                    <div><span class="connection-item-label">#${index + 1}</span> <span class="mono">${squareName(focusSquare)} -> ${item.square}</span></div>
                    <div>${fmtNumber(item.weight, 3)}</div>
                </div>
            `).join("")}
        </div>
    `;
}

function hideSamplePopover() {
    const node = document.getElementById("sample-popover");
    if (!node) return;
    node.innerHTML = "";
    node.classList.remove("is-visible");
}

function positionSamplePopover(event) {
    const node = document.getElementById("sample-popover");
    if (!node) return;
    const pad = 16;
    const width = Math.min(320, window.innerWidth - (pad * 2));
    const left = Math.min(window.innerWidth - width - pad, event.clientX + 18);
    const top = Math.min(window.innerHeight - node.offsetHeight - pad, event.clientY + 18);
    node.style.left = `${Math.max(pad, left)}px`;
    node.style.top = `${Math.max(pad, top)}px`;
}

function showSamplePopover(sample, focus, event) {
    const node = document.getElementById("sample-popover");
    if (!node || !sample || !Number.isInteger(Number(focus?.squareIndex))) return;
    const info = squareMetrics(sample, focus.squareIndex);
    const mapInfo = mapInfoForSample(sample);
    const mapKeys = mapKeysForSample(sample);
    const focusMeta = focus.mapName ? mapInfo[focus.mapName] : null;
    const focusedName = focus.mapName || mapKeys[0];
    const targetValue = Number(info.targetValues[focusedName] || 0);
    const predictionValue = Number(info.predictionValues[focusedName]);
    const predictionAvailable = Number.isFinite(predictionValue);
    node.innerHTML = `
        <div class="sample-popover-title mono">${info.square}</div>
        <div class="sample-popover-subtitle">${pieceName(info.piece)}</div>
        <div class="sample-popover-note">
            ${focusMeta
                ? `${focus.kind === "prediction" ? "Prediction" : "Target"} cell for ${focusMeta.title}.`
                : "Board square hover. Values below compare rule labels with model predictions for this square."}
        </div>
        <div class="sample-popover-grid">
            <div class="sample-popover-group">
                <div class="sample-popover-group-title">Focused Map</div>
                <div class="sample-popover-row">
                    <div class="sample-popover-key">Target</div>
                    <div class="sample-popover-value">${fmtNumber(targetValue, 3)}</div>
                </div>
                <div class="sample-popover-row">
                    <div class="sample-popover-key">Prediction</div>
                    <div class="sample-popover-value">${fmtMaybeNumber(predictionValue, 3)}</div>
                </div>
                <div class="sample-popover-row">
                    <div class="sample-popover-key">Abs error</div>
                    <div class="sample-popover-value">${predictionAvailable ? fmtNumber(Math.abs(targetValue - predictionValue), 3) : "n/a"}</div>
                </div>
            </div>
            <div class="sample-popover-group">
                <div class="sample-popover-group-title">All Maps</div>
                ${mapKeys.map((name) => `
                    <div class="sample-popover-row">
                        <div class="sample-popover-key">${mapInfo[name]?.title || name}</div>
                        <div class="sample-popover-value">${fmtNumber(info.targetValues[name], 3)} / ${fmtMaybeNumber(info.predictionValues[name], 3)}</div>
                    </div>
                `).join("")}
            </div>
        </div>
    `;
    node.classList.add("is-visible");
    positionSamplePopover(event);
}

function highlightFocusedSquare(squareIndex) {
    document.querySelectorAll(".sample-square.is-focus, .connection-board-square.is-focus").forEach((node) => {
        node.classList.remove("is-focus");
    });
    if (!Number.isInteger(squareIndex)) return;
    document.querySelectorAll(`[data-square-index="${squareIndex}"].sample-square, [data-square-index="${squareIndex}"].connection-board-square`).forEach((node) => {
        node.classList.add("is-focus");
    });
}

function bindSampleHover(sample) {
    const bind = (containerId) => {
        const node = document.getElementById(containerId);
        if (!node) return;
        node.onmousemove = (event) => {
            const target = event.target.closest("[data-square-index]");
            if (!target || !node.contains(target)) return;
            const squareIndex = Number(target.dataset.squareIndex || 0);
            activeConnectionSquareIndex = squareIndex;
            highlightFocusedSquare(squareIndex);
            renderConnectionLens(sample, squareIndex);
            showSamplePopover(sample, {
                squareIndex,
                mapName: target.dataset.mapName || "",
                kind: target.dataset.mapKind || "board",
            }, event);
        };
        node.onmouseleave = () => {
            hideSamplePopover();
        };
    };

    bind("sample-board");
    bind("geometry-compare");
}

function renderSample(sample) {
    activeConnectionSquareIndex = activeSquareForSample(sample);
    if (sample && sample.error) {
        setHtml("sample-summary", `
            <div class="info-note">
                Sample inspector unavailable: ${sample.error}
            </div>
        `);
        setHtml("sample-guide", "");
        setHtml("sample-board-label", "Sample unavailable");
        setHtml("sample-board", "");
        hideSamplePopover();
        setHtml("geometry-compare", "");
        setHtml("connection-lens", "");
        return;
    }

    if (!sample || !sample.piece_grid) {
        setHtml("sample-summary", '<div class="info-note">No geometry sample is available.</div>');
        setHtml("sample-guide", "");
        setHtml("sample-board-label", "No sample");
        setHtml("sample-board", "");
        hideSamplePopover();
        setHtml("geometry-compare", "");
        setHtml("connection-lens", "");
        return;
    }

    setHtml("sample-summary", renderSampleSummary(sample));
    setHtml("sample-guide", renderSampleGuide(sample));
    setHtml("sample-board-label", sample.board_view_label || "White at bottom");
    setHtml("sample-board", renderBoard(sample.piece_grid, activeConnectionSquareIndex));
    hideSamplePopover();
    renderGeometryCompare(sample);
    renderConnectionLens(sample, activeConnectionSquareIndex);
    bindSampleHover(sample);
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
