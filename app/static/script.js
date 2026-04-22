const modelSelect = document.getElementById("modelSelect");

modelSelect.addEventListener("change", () => {
    fetch("/set_model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: modelSelect.value })
    });
});

function toggleKP() {
    fetch("/toggle_kp", { method: "POST" });
}

function toggleCamera() {
    fetch('/toggle_camera', { method: 'POST' })
        .then(r => r.json())
        .then(d => {
            const video = document.getElementById('video');
            const overlay = document.getElementById('cameraOffOverlay');

            if (d.camera) {
                video.src = "/video_feed?" + Date.now();
                video.style.display = 'block';
                overlay.style.display = 'none';
            } else {
                video.src = "";
                video.removeAttribute("src");
                video.style.display = 'none';
                overlay.style.display = 'flex';
            }
        });
}


setInterval(() => {
    fetch("/stats")
        .then(r => r.json())
        .then(data => {
            document.getElementById("statusText").innerText = data.status;
            document.getElementById("probText").innerText = (data.prob * 100).toFixed(0) + "%";

            if (data.side === 'left') {
                document.getElementById("sideText").innerText = "Side: LEFT";
            } else if (data.side === 'right') {
                document.getElementById("sideText").innerText = "Side: RIGHT";
            } else {
                document.getElementById("sideText").innerText = "Side: —";
            }
        });
}, 500);