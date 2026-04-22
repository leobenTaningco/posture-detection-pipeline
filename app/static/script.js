// ── State ──────────────────────────────────────────────────────────────────
let camOn = false;   // matches data-active="false" on btnCamera in HTML
let kpOn = true;    // matches data-active="true"  on btnKP in HTML

let goodCount = 0, badCount = 0, totalCount = 0;
let goodStreak = 0, badStreak = 0;
let lastBadTime = null;
let fpsHistory = [];
let lastStatTime = performance.now();

// ── DOM refs ───────────────────────────────────────────────────────────────
const video = document.getElementById("video");
const overlay = document.getElementById("cameraOffOverlay");

// ── Camera helpers ─────────────────────────────────────────────────────────
function startCam() {
    video.src = "/video_feed?" + Date.now();
    video.style.display = "block";
    overlay.style.display = "none";
}

function stopCam() {
    video.src = "";
    video.style.display = "none";
    overlay.style.display = "flex";
}

// ── Toggle: Camera ─────────────────────────────────────────────────────────
function toggleCamera() {
    fetch("/toggle_camera", { method: "POST" })
        .then(r => r.json())
        .then(d => {
            camOn = d.camera;
            document.getElementById("btnCamera").dataset.active = camOn ? "true" : "false";
            if (camOn) startCam(); else stopCam();
        })
        .catch(err => console.error("toggleCamera failed:", err));
}

// ── Toggle: Keypoints ──────────────────────────────────────────────────────
function toggleKP() {
    fetch("/toggle_kp", { method: "POST" })
        .then(r => r.json())
        .then(d => {
            kpOn = d.draw_kp;
            document.getElementById("btnKP").dataset.active = kpOn ? "true" : "false";
        })
        .catch(err => console.error("toggleKP failed:", err));
}

// ── Model selector ─────────────────────────────────────────────────────────
function setModel(el) {
    document.querySelectorAll(".model-btn").forEach(b => b.classList.remove("active"));
    el.classList.add("active");
    fetch("/set_model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: el.dataset.model })
    }).catch(err => console.error("setModel failed:", err));
}

// ── Stats update ───────────────────────────────────────────────────────────
function update(d) {
    // FPS — derived from how often /stats actually responds
    const now = performance.now();
    const delta = (now - lastStatTime) / 1000;
    lastStatTime = now;
    if (delta > 0) {
        fpsHistory.push(1 / delta);
        if (fpsHistory.length > 10) fpsHistory.shift();
    }
    const fps = fpsHistory.length
        ? (fpsHistory.reduce((a, b) => a + b, 0) / fpsHistory.length).toFixed(1)
        : "—";
    document.getElementById("fpsVal").textContent = fps;
    document.getElementById("fpsBadge").textContent = fps + " FPS";

    // Posture label
    const label = document.getElementById("label");
    if (d.status === "good") {
        label.textContent = "GOOD";
        label.className = "posture-label good";
    } else if (d.status === "bad") {
        label.textContent = "BAD";
        label.className = "posture-label bad";
    } else {
        label.textContent = "—";
        label.className = "posture-label none";
    }

    // Confidence bar + text
    const pct = Math.round((d.prob || 0) * 100);
    const fill = document.getElementById("fill");
    fill.style.width = pct + "%";
    fill.style.background = d.status === "bad" ? "#ff4d4d" : "#7fff6e";
    document.getElementById("confPct").textContent = pct + "%";
    document.getElementById("confVal").textContent = pct + "%";

    // Side
    document.getElementById("side").textContent =
        d.side === "left" ? "LEFT" :
            d.side === "right" ? "RIGHT" : "—";

    // Session counters — only tick when a detection exists
    if (d.status !== "none") {
        totalCount++;
        document.getElementById("detections").textContent = totalCount;

        if (d.status === "good") {
            goodCount++;
            goodStreak++;
            badStreak = 0;
        } else {
            badCount++;
            badStreak++;
            goodStreak = 0;
            lastBadTime = new Date();
        }

        document.getElementById("goodStreak").textContent = goodStreak + " frames";
        document.getElementById("badStreak").textContent = badStreak + " frames";

        const pctGood = totalCount
            ? Math.round((goodCount / totalCount) * 100)
            : 0;
        document.getElementById("sessionGood").textContent = pctGood + "%";

        document.getElementById("lastCorrection").textContent =
            lastBadTime ? lastBadTime.toLocaleTimeString() : "—";
    }
}

// ── Poll /stats every 500ms ────────────────────────────────────────────────
setInterval(() => {
    fetch("/stats")
        .then(r => r.json())
        .then(update)
        .catch(() => { });
}, 500);

// ── Init: show camera-off state on load ────────────────────────────────────
stopCam();