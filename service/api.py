# service/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from redis import Redis
from rq import Queue
from rq.job import Job
import os
import sys

# Fix đường dẫn import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import REDIS_URL
from service.tasks import run_generation_task

app = FastAPI(title="SDXL Local API")

# Kết nối Redis
try:
    redis_conn = Redis.from_url(REDIS_URL)
    q = Queue(connection=redis_conn)
except Exception:
    print("❌ Warning: Redis chưa chạy. API sẽ lỗi khi gọi /generate")

class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = "low quality, blurry"

@app.post("/generate")
async def generate_image(req: GenerateRequest):
    """Nhận prompt và đẩy vào hàng đợi"""
    # Timeout 3600s (1 tiếng) vì chạy CPU rất lâu, tránh bị kill job giữa chừng
    job = q.enqueue(
        run_generation_task, 
        req.prompt, 
        req.negative_prompt,
        job_timeout=3600 
    )
    return {"job_id": job.get_id(), "status": "queued"}

@app.get("/status/{job_id}")
async def check_status(job_id: str):
    """Kiểm tra xem ảnh đã tạo xong chưa"""
    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except Exception:
        raise HTTPException(status_code=404, detail="Job ID not found")

    response = {
        "job_id": job_id,
        "status": job.get_status(), # queued, started, finished, failed
    }

    if job.is_finished:
        response["result"] = job.result  # Đường dẫn file ảnh
    elif job.is_failed:
        response["error"] = str(job.exc_info)

    return response