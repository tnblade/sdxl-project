# service/worker.py
import os
import sys
from redis import Redis
from rq import Worker, Queue

# --- CẤU HÌNH ĐƯỜNG DẪN ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(root_dir)
# --------------------------

from config import REDIS_URL

listen = ['default']

def start_worker():
    # Kết nối Redis
    try:
        conn = Redis.from_url(REDIS_URL)
        conn.ping() # Test kết nối
    except Exception as e:
        print(f"❌ Lỗi: Không thể kết nối Redis tại {REDIS_URL}")
        return

    print(f"👷 Worker started on Docker/CPU. Waiting for jobs...")

    # --- FIX LỖI Ở ĐÂY ---
    # Thay vì dùng 'with Connection(conn):', ta truyền connection trực tiếp vào Worker
    queues = [Queue(name, connection=conn) for name in listen]
    worker = Worker(queues, connection=conn)
    worker.work()

if __name__ == '__main__':
    start_worker()