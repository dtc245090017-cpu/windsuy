const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');
const statusEl = document.getElementById('status');

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.className = isError ? 'status error' : 'status';
}

function resizeCanvas() {
  overlay.width = video.clientWidth;
  overlay.height = video.clientHeight;
}

function drawBoxes(faces) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  const scaleX = overlay.width / video.videoWidth;
  const scaleY = overlay.height / video.videoHeight;
  faces.forEach(face => {
    const [x, y, w, h] = face.bbox;
    const rectX = x * scaleX;
    const rectY = y * scaleY;
    const rectW = w * scaleX;
    const rectH = h * scaleY;
    ctx.strokeStyle = '#00ff00';
    ctx.lineWidth = 2;
    ctx.strokeRect(rectX, rectY, rectW, rectH);
    const label = `ID ${face.person_id}: ${face.emotion} (${face.confidence.toFixed(2)})`;
    ctx.fillStyle = 'rgba(0,0,0,0.6)';
    ctx.fillRect(rectX, rectY - 18, ctx.measureText(label).width + 10, 18);
    ctx.fillStyle = '#00ff00';
    ctx.font = '12px monospace';
    ctx.fillText(label, rectX + 5, rectY - 5);
  });
}

async function pollFaces() {
  try {
    const res = await fetch('/api/frame');
    if (!res.ok) {
      const msg = await res.text();
      throw new Error(msg || 'Không lấy được dữ liệu frame');
    }
    const data = await res.json();
    drawBoxes(data.faces || []);
    setStatus('Đang chạy...');
  } catch (err) {
    console.error(err);
    setStatus(err.message || 'Lỗi camera hoặc server', true);
  }
}

function startPolling() {
  setInterval(pollFaces, 200); // 5 lần/giây
}

function startVideo() {
  video.src = '/video';
  video.addEventListener('loadedmetadata', () => {
    resizeCanvas();
    startPolling();
    setStatus('Đang chạy...');
  });
  video.addEventListener('error', () => {
    setStatus('Không mở được camera. Kiểm tra backend.', true);
  });
}

window.addEventListener('resize', resizeCanvas);
startVideo();
