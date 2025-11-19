# service/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from redis import Redis
from rq import Queue, Connection
from config import REDIS_URL
from service.tasks import run_generation_task
import uuid

app = FastAPI()

# Kết nối Redis
redis_conn = Redis.from_url(REDIS_URL)
q = Queue(connection=redis_conn)

# Định nghĩa gói tin gửi lên (Mở rộng thêm các trường mới)
class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = "low quality, blurry"
    width: int = 512
    height: int = 512
    steps: int = 20
    seed: int = -1
    # --- Trường mới ---
    image_path: str = None
    task_type: str = "txt2img" # txt2img, img2img, controlnet
    control_type: str = None
    strength: float = 0.7

@app.post("/generate")
def generate(req: GenerateRequest):
    job = q.enqueue(
        run_generation_task,
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        width=req.width,
        height=req.height,
        steps=req.steps,
        seed=req.seed,
        # Truyền tiếp cho Worker
        image_path=req.image_path,
        task_type=req.task_type,
        control_type=req.control_type,
        strength=req.strength
    )
    return {"job_id": job.get_id(), "status": "queued"}

@app.get("/status/{job_id}")
def get_status(job_id: str):
    job = q.fetch_job(job_id)
    if not job:
        return {"status": "unknown"}
    
    if job.is_failed:
        return {"status": "failed", "error": str(job.exc_info)}
    
    if job.is_finished:
        return {"status": "finished", "result": job.result}
        
    return {"status": job.get_status()}