from pathlib import Path
from typing import List

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .vision import VisionSystem

APP_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = APP_DIR.parent / "frontend"

app = FastAPI(title="Realtime Face Emotion")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vision = VisionSystem()


@app.on_event("shutdown")
async def shutdown_event():
    vision.shutdown()


@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="Frontend not found")
    return FileResponse(index_path)


@app.get("/video")
async def video_stream():
    if not vision.is_open():
        raise HTTPException(
            status_code=500,
            detail=(
                "Không mở được camera. Hãy kiểm tra CAMERA_INDEX trong README hoặc kết nối webcam."
            ),
        )

    def frame_generator():
        while True:
            try:
                frame_bytes, _ = vision.process_frame()
            except RuntimeError as exc:
                raise HTTPException(status_code=500, detail=str(exc))
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/frame")
async def api_frame():
    if not vision.is_open():
        raise HTTPException(
            status_code=500,
            detail=(
                "Không mở được camera. Hãy kiểm tra CAMERA_INDEX trong README hoặc kết nối webcam."
            ),
        )
    data: List[dict] = vision.get_latest_data()
    if not data:
        try:
            _, data = vision.process_frame()
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc))
    return JSONResponse(content={"faces": data})


app.mount(
    "/static",
    StaticFiles(directory=FRONTEND_DIR, html=True),
    name="static",
)


if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
