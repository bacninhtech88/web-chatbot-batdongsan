FROM python:3.11-slim

# Cài đặt các công cụ cần thiết, bao gồm Rust
RUN apt-get update && apt-get install -y curl gcc build-essential libssl-dev pkg-config \
    && curl https://sh.rustup.rs -sSf | sh -s -- -y \
    && export PATH="/root/.cargo/bin:$PATH" \
    && echo 'export PATH="/root/.cargo/bin:$PATH"' >> /etc/profile

# Cài poetry nếu bạn dùng, hoặc pip
ENV PATH="/root/.cargo/bin:$PATH"

# Tạo thư mục app
WORKDIR /app

# Copy trước requirements để tận dụng cache
COPY requirements.txt .

# Cài Python packages
RUN pip install --upgrade pip \
    && pip install --no-cache-dir maturin \
    && pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn vào
COPY . .

# Mở port
EXPOSE 8000

# Command chạy ứng dụng
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
