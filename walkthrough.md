# 🚀 Hướng dẫn deploy FastAPI AI Service lên HTTPS

## Những gì đã được sửa trong code

| File | Thay đổi |
|---|---|
| [`vector_service.py`](file:///d:/NamCuoi/TLCN/it-job-finder/it-job-finder-ai-chat/services/CV/storage/vector_service.py) | ✅ Thay ChromaDB → MongoDB Atlas Vector Search |
| [`core/config.py`](file:///d:/NamCuoi/TLCN/it-job-finder/it-job-finder-ai-chat/core/config.py) | ✅ Thêm CORS production URLs, xóa Chroma config |
| [`requirements.txt`](file:///d:/NamCuoi/TLCN/it-job-finder/it-job-finder-ai-chat/requirements.txt) | ✅ Xóa `chromadb==0.5.15` |
| [`render.yaml`](file:///d:/NamCuoi/TLCN/it-job-finder/it-job-finder-ai-chat/render.yaml) | ✅ Tạo mới — config FastAPI web + Celery worker |
| [`it-job-finder-server/.env`](file:///d:/NamCuoi/TLCN/it-job-finder/it-job-finder-server/.env) | ✅ `FASTAPI_URL=https://it-job-finder-ai.onrender.com` |

---

## Bước 1: Tạo MongoDB Atlas cluster mới (miễn phí)

> **Cluster này dùng cho**: `cv_metadata`, `chat_history`, `cv_vectors` (vector embeddings)

1. Vào [cloud.mongodb.com](https://cloud.mongodb.com) → đăng nhập
2. Click **"Create"** → chọn **"M0 Free"** → Region: **Singapore (AP)** → Create Cluster
3. Đặt tên: `cv-chatbot-cluster`
4. **Database Access** → Add new user:
   - Username: `cv_chatbot_user`
   - Password: tạo auto-generate → **copy lại**
   - Role: `Read and write to any database`
5. **Network Access** → Add IP: `0.0.0.0/0` (allow all — cần cho Render)
6. **Connect** → Drivers → Copy connection string:
   ```
   mongodb+srv://cv_chatbot_user:<password>@cv-chatbot-cluster.xxxxx.mongodb.net
   ```
   → Thay `<password>` bằng password vừa tạo → **lưu lại chuỗi này**

---

## Bước 2: Tạo Atlas Vector Search Index

> Bước này rất quan trọng — phải làm TRƯỚC khi test chat

1. Vào cluster vừa tạo → Tab **"Atlas Search"** (hoặc "Search")
2. Click **"Create Search Index"**
3. Chọn **"Atlas Vector Search"** → **"JSON Editor"**
4. Database: `cv_chatbot`, Collection: `cv_vectors`
5. Dán JSON sau vào:

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 384,
      "similarity": "cosine"
    },
    {
      "type": "filter",
      "path": "cv_id"
    },
    {
      "type": "filter",
      "path": "user_id"
    }
  ]
}
```

6. Index Name: `cv_embedding_index`
7. Click **"Create Search Index"** → đợi ~2-3 phút để build

---

## Bước 3: Push code lên GitHub

```bash
cd d:\NamCuoi\TLCN\it-job-finder\it-job-finder-ai-chat
git add .
git commit -m "feat: migrate from Celery to FastAPI BackgroundTasks, remove Redis dependency"
git push origin master
```

> [!IMPORTANT]
> Đảm bảo file `.env` **KHÔNG** được commit lên GitHub (kiểm tra `.gitignore`).

---

## Bước 4: Deploy lên Render

1. Vào [render.com](https://render.com) → đăng nhập → **"New"** → **"Blueprint"**
2. Kết nối GitHub → chọn repo `it-job-finder-ai-chat`
3. Render sẽ tự đọc `render.yaml` và hiện ra **1 service duy nhất**:
   - `it-job-finder-ai` (Web Service — FastAPI)
4. Click **"Apply"**

---

## Bước 5: Set Environment Variables trên Render

Vào service `it-job-finder-ai` → **Environment** → Add variables hoặc chỉnh sửa (chỉ cần cấu hình một lần):

| Key | Value |
|---|---|
| `ACCESS_TOKEN_SECRET` | `bb30dacf285e732a6b936aa363bf6585b763218a770921cda8a88de1a45189450d699c82121737b1ac2d446861a1d78e7134dcdc5d22583375db357d7277c571` |
| `MONGODB_URI` | `mongodb+srv://a01654464356_db_user:1SI0Yi35yYnYzUrz@cv-chatbot-cluster.wyrxasp.mongodb.net/?appName=cv-chatbot-cluster` |
| `MONGODB_DB` | `cv_chatbot` |
| `JOBS_MONGO_URI` | URI Atlas cluster của Node.js server (chuỗi trong `it-job-finder-server/.env` dòng 2) |
| `JOBS_MONGO_DB` | `ITJOBS` |
| `GEMINI_API_KEY` | API key Gemini của bạn |
| `GROQ_API_KEY` | API key Groq của bạn |

> [!IMPORTANT]
> **Đảm bảo Whitelist IP trên MongoDB Atlas:**
> Để Render có thể kết nối thành công với MongoDB Atlas mà không bị lỗi SSL handshake, bạn **bắt buộc** phải cấu hình Network Access trên MongoDB Atlas cho phép mọi IP (`0.0.0.0/0`).
> Xem hướng dẫn chi tiết ở phần phản hồi chat để khắc phục lỗi SSL này.

---

## Bước 6: Kiểm tra URL sau khi deploy

Sau khi deploy xong, Render sẽ cấp URL:
- FastAPI: `https://it-job-finder-ai.onrender.com`

**Test thử**:
```
https://it-job-finder-ai.onrender.com/health
https://it-job-finder-ai.onrender.com/docs
```

---

## Bước 7: Cập nhật FASTAPI_URL trên Render (Node.js server)

Vào Render → service `it-job-finder-server` → **Environment** → thêm/sửa:
```
FASTAPI_URL = https://it-job-finder-ai.onrender.com
```
→ Save → service tự động redeploy.

---

## ✅ Kết quả sau khi hoàn thành

```
User → https://it-job-finder-client-five.vercel.app (Vercel, đã chạy)
          → https://it-job-finder-server.onrender.com (Render, đã chạy)
                → https://it-job-finder-ai.onrender.com (Render FastAPI Web Service + BackgroundTasks)
                      → MongoDB Atlas cv-chatbot-cluster (chat, metadata, vectors)
                      → MongoDB Atlas ITJOBS cluster (jobs data, đã có)
```

---

## Trả lời câu hỏi của bạn

> **"Sau khi deploy xong thì có dùng được như Docker không?"**

**Đúng!** Sau khi setup xong:
- Không cần chạy Docker nữa.
- Không cần mở máy tính lên.
- Hệ thống chạy 24/7 trên cloud, có HTTPS, có domain public.

> **"Push code lên master là tự động chạy?"**

**Đúng!** Khi đã kết nối GitHub với Render:
- Push lên `master` (hoặc `main`) → Render **tự động detect** → build lại → deploy mới.
- Vercel cũng vậy — frontend push là tự redeploy.

> [!WARNING]
> **Giới hạn Render Free Tier**: Sau 15 phút không có request, service sẽ "ngủ". Request đầu tiên sau khi ngủ mất ~30-50 giây để wake up. Đây là hạn chế của free plan, không thể tránh khỏi nếu không nâng cấp.
