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
}

async function animateSteps() {
    for (let i = 0; i < 3; i++) {
        setStep(i + 1, "active");
        await sleep(900);
        setStep(i + 1, "done");
    }
    setStep(4, "active");
}
const sleep = ms => new Promise(r => setTimeout(r, ms));

// ── Fix #10: stable DOM helpers — never replace elements via outerHTML ──
// All text/content updates go through these helpers so element references
// stay valid across multiple translation runs.

function setImageSlot(id, src, alt) {
    const wrap = document.getElementById(id);
    if (src) {
        wrap.innerHTML = `<img src="${src}" alt="${escHtml(alt)}">`;
    } else {
        wrap.innerHTML = `<div class="image-placeholder">No image yet</div>`;
    }
}

function setTextSlot(wrapperId, text) {
    const wrap = document.getElementById(wrapperId);
    if (text) {
        // Switch to <pre> styling via class, update content safely
        wrap.className = "text-content";
        // Use a real <pre> inside the stable wrapper
        wrap.innerHTML = `<pre style="padding:16px;font-size:0.78rem;line-height:1.7;
            color:var(--text);white-space:pre-wrap;word-break:break-word;
            min-height:80px;font-family:var(--mono);margin:0">${escHtml(text)}</pre>`;
    } else {
        wrap.className = "empty-text";
        wrap.textContent = wrapperId === "extractedWrap"
            ? "Extracted text will appear here."
            : "Translated text will appear here.";
    }
}

// ── Main upload ─────────────────────────────────────────────────
async function runTranslation() {
    const fileInput = document.getElementById("imageInput");
    if (!fileInput.files.length) return;

    const btn = document.getElementById("translateBtn");
    btn.disabled = true;
    btn.textContent = "Processing…";
    hideError(); hideWarning(); resetSteps(); clearOutputs();

    const stepAnim = animateSteps();
    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    let data;
    try {
        const response = await fetch("/upload", { method: "POST", body: formData });
        await stepAnim;

        if (!response.ok) {
            const err = await response.json().catch(() => ({ error: `Server error ${response.status}` }));

            // ── Fix #5: show original image even when server returns an error ──
            if (err.original_image) {
                setImageSlot("originalWrap", `/${err.original_image}`, "Original");
                document.getElementById("dlOriginal").style.display = "flex";
            }

            showError(err.detail || err.error || `Unexpected error (${response.status})`);
            resetSteps();
            return;
        }
        data = await response.json();
    } catch (e) {
        await stepAnim;
        showError("Could not reach the server. Is the backend running?  uvicorn backend.main:app --reload");
        resetSteps(); return;
    } finally {
        btn.disabled = false;
        btn.textContent = "Translate Image";
    }

    setStep(4, "done");
    currentData = data;

    if (data.warning) showWarning(data.warning);

    if (data.detected_language && data.detected_language !== "unknown") {
        document.getElementById("langBadge").style.display = "block";
        document.getElementById("langName").textContent = data.detected_language.toUpperCase();
    }

    // Images
    if (data.original_image) {
        setImageSlot("originalWrap", `/${data.original_image}`, "Original");
        document.getElementById("dlOriginal").style.display = "flex";
    }
    if (data.translated_image) {
        setImageSlot("translatedWrap", `/${data.translated_image}`, "Translated overlay");
        document.getElementById("dlTranslated").style.display = "flex";
    }

    // Text — Fix #10: use stable wrapper, never outerHTML swap
    setTextSlot("extractedWrap",     data.extracted_text    || "");
    setTextSlot("translatedTextWrap", data.translated_text  || "");

    if (data.extracted_text)   document.getElementById("dlExtracted").style.display    = "flex";
    if (data.translated_text)  document.getElementById("dlTranslatedTxt").style.display = "flex";
}

// ── Utilities ───────────────────────────────────────────────────
function escHtml(str) {
    return String(str)
        .replace(/&/g,"&amp;")
        .replace(/</g,"&lt;")
        .replace(/>/g,"&gt;")
        .replace(/"/g,"&quot;");
}

function clearOutputs() {
    setImageSlot("originalWrap",   null, "");
    setImageSlot("translatedWrap", null, "");
    setTextSlot("extractedWrap",     "");
    setTextSlot("translatedTextWrap","");

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

    // Get text from the inner <pre> if present, else the wrapper itself
    const pre  = wrap.querySelector("pre");
    const text = (pre || wrap).textContent || "";
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
    const pre       = wrap.querySelector("pre");
    const text      = (pre || wrap).textContent || "";
    if (!text) return;
    const blob = new Blob([text], { type: "text/plain" });
    const a    = document.createElement("a");
    a.href     = URL.createObjectURL(blob);
    a.download = which === "extracted" ? "extracted.txt" : "translated.txt";
    document.body.appendChild(a); a.click();
    document.body.removeChild(a); URL.revokeObjectURL(a.href);
}
