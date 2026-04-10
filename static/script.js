const MAX_SIZE = 5 * 1024 * 1024;
let currentData = null;

// ── File selection ──────────────────────────────────────────────
const input = document.getElementById("imageInput");
const zone  = document.getElementById("uploadZone");

input.addEventListener("change", () => handleFileSelect(input.files[0]));

zone.addEventListener("dragover",  e => { e.preventDefault(); zone.classList.add("drag-over"); });
zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));
zone.addEventListener("drop", e => {
    e.preventDefault();
    zone.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file) { input.files = e.dataTransfer.files; handleFileSelect(file); }
});

function handleFileSelect(file) {
    if (!file) return;
    const warn = document.getElementById("sizeWarning");
    const btn  = document.getElementById("translateBtn");
    if (file.size > MAX_SIZE) {
        warn.style.display = "block";
        btn.disabled = true;
        document.getElementById("fileName").textContent = "";
        return;
    }
    warn.style.display = "none";
    btn.disabled = false;
    document.getElementById("fileName").textContent = `✓ ${file.name}`;
    hideError(); resetSteps();
}

// ── Step helpers ────────────────────────────────────────────────
function setStep(n, state) {
    const el   = document.getElementById(`step${n}`);
    const icon = document.getElementById(`step${n}icon`);
    el.classList.remove("active", "done");
    if (state === "active") el.classList.add("active");
    if (state === "done")   { el.classList.add("done"); icon.textContent = "✓"; }
}

function resetSteps() {
    [1,2,3,4].forEach(n => {
        const el   = document.getElementById(`step${n}`);
        const icon = document.getElementById(`step${n}icon`);
        el.classList.remove("active","done");
        icon.textContent = String(n);
    });
    // FIX #2: Always clear the interval on reset — prevents memory leak
    if (window._renderProgressInterval) {
        clearInterval(window._renderProgressInterval);
        window._renderProgressInterval = null;
    }
    const bar     = document.getElementById("step4bar");
    const phase   = document.getElementById("step4phase");
    const counter = document.getElementById("step4counter");
    const prog    = document.getElementById("step4progress");
    if (bar)     { bar.style.width = "0%"; bar.style.background = "var(--accent)"; }
    if (phase)   phase.textContent = "Rendering...";
    if (counter) counter.textContent = "";
    if (prog)    prog.style.display = "none";
}

// FIX #2: animateSteps() no longer polls window._renderDone.
// It simply animates steps 1-3, activates step 4, then RETURNS.
// Step 4 completion is driven by runTranslation() after data arrives.
async function animateSteps() {
    for (let i = 0; i < 3; i++) {
        setStep(i + 1, "active");
        await sleep(900);
        setStep(i + 1, "done");
    }
    setStep(4, "active");
    // Show step 4 progress bar immediately
    const prog = document.getElementById("step4progress");
    if (prog) prog.style.display = "block";
}

function startRenderProgress() {
    const bar      = document.getElementById("step4bar");
    const phase    = document.getElementById("step4phase");
    const counter  = document.getElementById("step4counter");
    const icon     = document.getElementById("step4icon");
    const progress = document.getElementById("step4progress");
    if (!progress) return;

    const total = 96;
    const phases = [
        { pct: 0,  label: "Sampling backgrounds..." },
        { pct: 25, label: "Rendering regions..." },
        { pct: 60, label: "Compositing overlay..." },
        { pct: 85, label: "Saving output image..." },
    ];

    let pct = 0;
    progress.style.display = "block";

    if (icon) icon.innerHTML = `<svg width="14" height="14" viewBox="0 0 14 14" style="animation:spin4 1s linear infinite;display:block"><circle cx="7" cy="7" r="5" stroke="currentColor" stroke-width="1.5" fill="none" stroke-dasharray="20" stroke-dashoffset="8" stroke-linecap="round"/></svg>`;

    // FIX #2: Store interval ref so resetSteps() can always clear it
    window._renderProgressInterval = setInterval(() => {
        pct += Math.random() * 6 + 2;
        if (pct >= 95) pct = 95;
        const done = Math.round((pct / 100) * total);
        bar.style.width = pct.toFixed(0) + "%";
        const currentPhase = [...phases].reverse().find(p => pct >= p.pct);
        if (currentPhase) phase.textContent = currentPhase.label;
        counter.textContent = done + " / " + total + " done";
    }, 200);
}

function completeRenderProgress() {
    if (window._renderProgressInterval) {
        clearInterval(window._renderProgressInterval);
        window._renderProgressInterval = null;
    }
    const bar     = document.getElementById("step4bar");
    const phase   = document.getElementById("step4phase");
    const counter = document.getElementById("step4counter");
    if (bar)     { bar.style.width = "100%"; bar.style.background = "var(--accent)"; }
    if (phase)   phase.textContent = "Complete";
    if (counter) counter.textContent = "100%";
}

const sleep = ms => new Promise(r => setTimeout(r, ms));

// ── DOM helpers ─────────────────────────────────────────────────
function setImageSlot(id, src, alt) {
    const wrap = document.getElementById(id);
    if (src) {
        wrap.innerHTML = `<img src="${src}" alt="${escHtml(alt)}">`;
    } else {
        wrap.innerHTML = `<div class="image-placeholder">No image yet</div>`;
    }
}

function markdownToHtml(text) {
    const lines = text.split("\n").map(l => l.trim());
    let html = "";
    let inTable = false;
    let tableHtml = "";

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();
        if (line.startsWith("|")) {
            const cells = line.slice(1, -1).split("|").map(c => c.trim());
            const isSeparator = cells.every(c => /^[-: ]+$/.test(c));
            if (!inTable) {
                inTable = true;
                tableHtml = `<table style="border-collapse:collapse;width:100%;font-size:0.78rem;margin:8px 0">`;
                tableHtml += "<thead><tr>" + cells.map(c =>
                    `<th style="border:1px solid var(--border);padding:6px 10px;background:var(--surface2);text-align:left;white-space:nowrap">${escHtml(c)}</th>`
                ).join("") + "</tr></thead><tbody>";
            } else if (isSeparator) {
                continue;
            } else {
                tableHtml += "<tr>" + cells.map(c =>
                    `<td style="border:1px solid var(--border);padding:5px 10px;white-space:nowrap">${escHtml(c)}</td>`
                ).join("") + "</tr>";
            }
        } else {
            if (inTable) {
                tableHtml += "</tbody></table>";
                html += tableHtml;
                tableHtml = "";
                inTable = false;
            }
            if (line === "") {
                html += "<br>";
            } else {
                html += `<div style="padding:1px 16px;font-size:0.78rem;line-height:1.7;color:var(--text)">${escHtml(line)}</div>`;
            }
        }
    }
    if (inTable) { tableHtml += "</tbody></table>"; html += tableHtml; }
    return html;
}

function setTextSlot(wrapperId, text) {
    const wrap = document.getElementById(wrapperId);
    if (text) {
        wrap.className = "text-content";
        const hasTable = text.includes("|");
        if (hasTable) {
            wrap.innerHTML = `<div style="padding:8px 0;overflow-x:auto">${markdownToHtml(text)}</div>`;
        } else {
            wrap.innerHTML = `<pre style="padding:16px;font-size:0.78rem;line-height:1.7;color:var(--text);white-space:pre-wrap;word-break:break-word;min-height:80px;font-family:var(--mono);margin:0">${escHtml(text)}</pre>`;
        }
    } else {
        wrap.className = "empty-text";
        wrap.textContent = wrapperId === "extractedWrap"
            ? "Extracted text will appear here."
            : "Translated text will appear here.";
    }
}

// ── SSE fetch helper (unused in /upload path, kept for /upload-stream) ──
async function fetchSSE(url, formData, { onProgress, onResult, onError }) {
    const response = await fetch(url, { method: "POST", body: formData });
    if (!response.ok) {
        const err = await response.json().catch(() => ({ error: `Server error ${response.status}` }));
        onError(err);
        return;
    }
    const reader  = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer    = "";
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop();
        for (const part of parts) {
            if (!part.trim() || part.startsWith(":")) continue;
            const lines     = part.split("\n");
            const eventLine = lines.find(l => l.startsWith("event:"));
            const dataLine  = lines.find(l => l.startsWith("data:"));
            if (!dataLine) continue;
            const eventType = eventLine ? eventLine.slice(6).trim() : "message";
            let payload;
            try { payload = JSON.parse(dataLine.slice(5).trim()); }
            catch { continue; }
            if (eventType === "progress") onProgress(payload.done, payload.total);
            else if (eventType === "result") { onResult(payload); return; }
            else if (eventType === "error")  { onError(payload); return; }
        }
    }
}

function updateRenderProgress(done, total) {
    const bar      = document.getElementById("step4bar");
    const phase    = document.getElementById("step4phase");
    const counter  = document.getElementById("step4counter");
    const progress = document.getElementById("step4progress");
    if (!bar) return;
    if (progress) progress.style.display = "block";
    const pct = total > 0 ? Math.round((done / total) * 100) : 0;
    bar.style.width = pct + "%";
    counter.textContent = done + " / " + total + " done";
    const phases = [
        { pct: 0,  label: "Sampling backgrounds..." },
        { pct: 25, label: "Rendering regions..." },
        { pct: 60, label: "Compositing overlay..." },
        { pct: 85, label: "Saving output image..." },
    ];
    const current = [...phases].reverse().find(p => pct >= p.pct);
    if (current && phase) phase.textContent = current.label;
}

// ── Main upload ─────────────────────────────────────────────────
// FIX #1: Use Promise.all() to run animation and fetch concurrently.
// The fetch .json() body is consumed IMMEDIATELY inside the Promise chain,
// never deferred past an await boundary — eliminating the race condition.
async function runTranslation() {
    const fileInput = document.getElementById("imageInput");
    if (!fileInput.files.length) return;

    const btn = document.getElementById("translateBtn");
    btn.disabled    = true;
    btn.textContent = "Processing…";
    hideError(); hideWarning(); resetSteps(); clearOutputs();

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    // Start fake progress bar for step 4 alongside animation
    // (will be overridden by real data once fetch resolves)
    setTimeout(() => startRenderProgress(), 2700); // ~when steps 1-3 finish

    // FIX #1: Both promises run concurrently.
    // fetch chain reads .json() eagerly — never deferred past an await.
    const [, result] = await Promise.all([
        animateSteps(),
        fetch("/upload", { method: "POST", body: formData })
            .then(async (response) => {
                // ← Body consumed immediately here, inside the Promise chain
                const data = await response.json().catch(() => ({
                    error: `Server error ${response.status}`,
                }));
                // Attach HTTP meta so we can inspect it after Promise.all
                data.__ok     = response.ok;
                data.__status = response.status;
                return data;
            })
            .catch((networkErr) => ({
                // Network-level failure (server down, CORS, etc.)
                error: "Could not reach the server. Is the backend running?  uvicorn backend.main:app --reload",
                __ok:     false,
                __status: 0,
            })),
    ]);

    btn.disabled    = false;
    btn.textContent = "Translate Image";

    // ── Error path ──────────────────────────────────────────────
    if (!result.__ok) {
        if (result.original_image) {
            setImageSlot("originalWrap", `/${result.original_image}`, "Original");
            document.getElementById("dlOriginal").style.display = "flex";
        }
        showError(result.detail || result.error || `Unexpected error (${result.__status})`);
        resetSteps();
        return;
    }

    // ── Success path ────────────────────────────────────────────
    completeRenderProgress();
    await sleep(400);
    setStep(4, "done");

    // Strip internal meta-fields before storing in module-level currentData
    const { __ok, __status, ...cleanResult } = result;
    currentData = cleanResult;

    if (result.warning) showWarning(result.warning);

    if (result.detected_language && result.detected_language !== "unknown") {
        document.getElementById("langBadge").style.display = "block";
        document.getElementById("langName").textContent =
            result.detected_language.toUpperCase();
    }

    // Images
    if (result.original_image) {
        setImageSlot("originalWrap", `/${result.original_image}`, "Original");
        document.getElementById("dlOriginal").style.display = "flex";
    }
    if (result.translated_image) {
        setImageSlot("translatedWrap", `/${result.translated_image}`, "Translated overlay");
        document.getElementById("dlTranslated").style.display = "flex";
    }

    // Text
    setTextSlot("extractedWrap",      result.extracted_text  || "");
    setTextSlot("translatedTextWrap", result.translated_text || "");

    if (result.extracted_text)
        document.getElementById("dlExtracted").style.display     = "flex";
    if (result.translated_text)
        document.getElementById("dlTranslatedTxt").style.display = "flex";
}

// ── Utilities ───────────────────────────────────────────────────
function escHtml(str) {
    return String(str)
        .replace(/&/g,"&amp;").replace(/</g,"&lt;")
        .replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

function clearOutputs() {
    setImageSlot("originalWrap",   null, "");
    setImageSlot("translatedWrap", null, "");
    setTextSlot("extractedWrap",      "");
    setTextSlot("translatedTextWrap", "");
    ["dlOriginal","dlTranslated","dlExtracted","dlTranslatedTxt"]
        .forEach(id => { document.getElementById(id).style.display = "none"; });
    document.getElementById("langBadge").style.display = "none";
    currentData = null;
}

function showError(msg)   { const b = document.getElementById("errorBox");   b.style.display="block"; b.textContent=`⚠ ${msg}`; }
function hideError()      { document.getElementById("errorBox").style.display="none"; }
function showWarning(msg) { const b = document.getElementById("warningBar"); b.style.display="block"; b.textContent=`ℹ ${msg}`; }
function hideWarning()    { document.getElementById("warningBar").style.display="none"; }

// ── Copy ────────────────────────────────────────────────────────
async function copyText(which) {
    const wrapperId = which === "extracted" ? "extractedWrap" : "translatedTextWrap";
    const btnId     = which === "extracted" ? "copyExtracted" : "copyTranslated";
    const btn       = document.getElementById(btnId);
    const wrap      = document.getElementById(wrapperId);
    const pre       = wrap.querySelector("pre");
    const text      = (pre || wrap).textContent || "";
    if (!text || wrap.classList.contains("empty-text")) return;
    try {
        await navigator.clipboard.writeText(text);
        const orig = btn.textContent;
        btn.textContent = "✓ Copied"; btn.classList.add("copied");
        setTimeout(() => { btn.textContent = orig; btn.classList.remove("copied"); }, 1800);
    } catch { btn.textContent = "Failed"; }
}

// ── Download ────────────────────────────────────────────────────
function downloadFile(which) {
    if (!currentData) return;
    const path = which === "original" ? currentData.original_image : currentData.translated_image;
    if (!path) return;
    const a = document.createElement("a");
    a.href = `/${path}`;
    a.download = which === "original" ? "original.jpg" : "translated_overlay.jpg";
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
}

function downloadText(which) {
    const wrapperId = which === "extracted" ? "extractedWrap" : "translatedTextWrap";
    const wrap      = document.getElementById(wrapperId);
    let text        = "";
    const table     = wrap.querySelector("table");
    if (table) {
        const rows = table.querySelectorAll("tr");
        const colWidths = [];
        rows.forEach(row => {
            row.querySelectorAll("th, td").forEach((cell, i) => {
                colWidths[i] = Math.max(colWidths[i] || 0, cell.textContent.trim().length);
            });
        });
        const lines = [];
        rows.forEach((row, ri) => {
            const cells = row.querySelectorAll("th, td");
            lines.push(Array.from(cells).map((c, i) =>
                c.textContent.trim().padEnd(colWidths[i])
            ).join("  |  "));
            if (ri === 0) lines.push(colWidths.map(w => "-".repeat(w)).join("--+--"));
        });
        const divs    = wrap.querySelectorAll("div > div");
        const preText = Array.from(divs).map(d => d.textContent.trim()).filter(Boolean).join("\n");
        text = (preText ? preText + "\n\n" : "") + lines.join("\n");
    } else {
        const pre = wrap.querySelector("pre");
        text = (pre || wrap).textContent || "";
    }
    if (!text.trim()) return;
    const blob = new Blob([text], { type: "text/plain" });
    const a    = document.createElement("a");
    a.href     = URL.createObjectURL(blob);
    a.download = which === "extracted" ? "extracted.txt" : "translated.txt";
    document.body.appendChild(a); a.click();
    document.body.removeChild(a); URL.revokeObjectURL(a.href);
}