# Realtime Multi-face Emotion Recognition (FastAPI + MediaPipe + OpenCV)

Dự án mẫu chạy offline trên Windows để nhận diện nhiều khuôn mặt, tracking ID và đo biểu cảm realtime.

## Chuẩn bị môi trường (Windows)
1. Mở PowerShell tại thư mục dự án.
2. Tạo và kích hoạt môi trường ảo:
   ```powershell
   python -m venv venv
   venv\Scripts\activate
   ```
3. Cài đặt phụ thuộc:
   ```powershell
   pip install -r requirements.txt
   ```

## Chạy server
```powershell
uvicorn backend.main:app --reload
```
Sau khi chạy, mở trình duyệt tại http://127.0.0.1:8000

Nếu không mở được webcam, thử chỉnh `CAMERA_INDEX` trong `backend/vision.py` (tham số mặc định 0). Thay bằng 1, 2... nếu đang dùng nhiều thiết bị camera.

## Cấu trúc
```
backend/
  main.py        - FastAPI server, MJPEG stream và API JSON
  vision.py      - MediaPipe detection, centroid tracking, emotion (FER/stub)
  tracker.py     - Centroid tracker tự viết
frontend/
  index.html     - Trang chính
  app.js         - Gọi API, vẽ bbox/label
  style.css      - Giao diện
logs/emotions.jsonl - Log kết quả cảm xúc (tự tạo)
faces_db/          - Thư mục trống sẵn dùng mở rộng lưu khuôn mặt
requirements.txt   - Danh sách thư viện
```

## API
- `GET /` : Trang web hiển thị video + overlay.
- `GET /video` : MJPEG stream từ webcam.
- `GET /api/frame` : JSON realtime danh sách khuôn mặt `{person_id, bbox, emotion, confidence}`.

## Ghi log
Mỗi khi chạy suy luận cảm xúc (mặc định 5 frame/lần), server ghi một dòng JSON vào `logs/emotions.jsonl`:
```json
{"ts": 1710000000.0, "person_id": 0, "emotion": "happy", "confidence": 0.92}
```

## Emotion model & fallback
- Mặc định dùng thư viện [FER](https://github.com/justinshenk/fer) (Keras/TensorFlow).
- Nếu FER hoặc TensorFlow lỗi, code tự động fallback về model stub trả "neutral" với confidence 0.0 để vẫn chạy được. Bạn có thể thay stub bằng DeepFace hoặc model riêng bằng cách sửa `EmotionRecognizer` trong `backend/vision.py`.

## Lưu ý khi chạy
- Ứng dụng chạy hoàn toàn offline, không gọi cloud API.
- Nếu hiệu năng thấp, giảm kích thước frame camera hoặc tăng `emotion_every` để suy luận cảm xúc ít hơn.
- Đảm bảo đã cấp quyền cho webcam trên Windows.
