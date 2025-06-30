# 使用較新的 Python base image（含編譯工具）
FROM python:3.10-slim

# 安裝必要套件（含 git, cmake, build-essential）
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 建立 app 資料夾
WORKDIR /app

# 複製 requirements 並安裝
COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install sentencepiece

# 安裝最新版 llama.cpp
RUN pip install llama_cpp_python==0.3.9

# 複製 app 原始碼
COPY ./app /app

# expose port
EXPOSE 8000

# 啟動 FastAPI 應用
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]