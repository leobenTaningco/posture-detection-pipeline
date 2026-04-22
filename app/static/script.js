let camOn = false;
let kpOn = true;

let goodCount = 0, badCount = 0, totalCount = 0;
let goodStreak = 0, badStreak = 0;
let lastBadTime = null;
let fpsHistory = [];
let lastStatTime = performance.now();

let alertActive = false;
let lastBeepTime = 0;
const BAD_THRESHOLD_SEC = 10;
const BEEP_INTERVAL_MS = 3000;

const video = document.getElementById("video");
const overlay = document.getElementById("cameraOffOverlay");
const alertBanner = document.getElementById("alertBanner");

function beep() {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const oscillator = ctx.createOscillator();
    const gain = ctx.createGain();
    oscillator.connect(gain);
    gain.connect(ctx.destination);
    oscillator.type = "sine";
    oscillator.frequency.setValueAtTime(880, ctx.currentTime);
    gain.gain.setValueAtTime(0.4, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.4);
    oscillator.start(ctx.currentTime);
    oscillator.stop(ctx.currentTime + 0.4);
}

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

function toggleKP() {
    fetch("/toggle_kp", { method: "POST" })
        .then(r => r.json())
        .then(d => {
            kpOn = d.draw_kp;
            document.getElementById("btnKP").dataset.active = kpOn ? "true" : "false";
        })
        .catch(err => console.error("toggleKP failed:", err));
}

function setModel(el) {
    document.querySelectorAll(".model-btn").forEach(b => b.classList.remove("active"));
    el.classList.add("active");
    fetch("/set_model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: el.dataset.model })
    }).catch(err => console.error("setModel failed:", err));
}

function update(d) {
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

    const pct = Math.round((d.prob || 0) * 100);
    const fill = document.getElementById("fill");
    fill.style.width = pct + "%";
    fill.style.background = d.status === "bad" ? "#ff4d4d" : "#7fff6e";
    document.getElementById("confPct").textContent = pct + "%";
    document.getElementById("confVal").textContent = pct + "%";

    document.getElementById("side").textContent =
        d.side === "left" ? "LEFT" :
            d.side === "right" ? "RIGHT" : "—";

    const badDuration = d.bad_duration || 0;
    document.getElementById("badDuration").textContent =
        d.status === "bad" ? badDuration.toFixed(1) + "s" : "—";

    if (d.status === "bad" && badDuration >= BAD_THRESHOLD_SEC) {
        if (!alertActive) {
            alertActive = true;
            alertBanner.classList.add("visible");
        }
        const nowMs = Date.now();
        if (nowMs - lastBeepTime > BEEP_INTERVAL_MS) {
            beep();
            lastBeepTime = nowMs;
        }
    } else {
        if (alertActive) {
            alertActive = false;
            alertBanner.classList.remove("visible");
        }
    }

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

setInterval(() => {
    fetch("/stats")
        .then(r => r.json())
        .then(update)
        .catch(() => { });
}, 500);

stopCam();