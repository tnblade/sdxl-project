# Sử dụng Python 3.10
FROM python:3.10-slim

# 1. Cài đặt các thư viện hệ thống
# ĐÃ SỬA: Thay libgl1-mesa-glx bằng libgl1
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. Thiết lập thư mục làm việc
WORKDIR /app

# 3. Copy file requirements và cài đặt thư viện
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 4. Copy toàn bộ code vào trong ảnh
COPY . .

# 5. Mở cổng
EXPOSE 8000 8501

# Lệnh mặc định
CMD ["uvicorn", "service.api:app", "--host", "0.0.0.0", "--port", "8000"]