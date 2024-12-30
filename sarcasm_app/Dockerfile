# Базовый образ
FROM python:3.10-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gdal-bin \
    libgdal-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Установка рабочей директории
WORKDIR /app

# Копирование файлов приложения
COPY requirements.txt ./requirements.txt
COPY script.py ./script.py

# Копирование модели
COPY HybridModel_state_dict.pth ./HybridModel_state_dict.pth

# Копирование токенизатора
COPY HybridTokenizer/ ./HybridTokenizer/

# Копирование README
COPY README.md ./README.md

# Обновление pip
RUN pip install --upgrade pip

# Установка PyTorch с CUDA
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Установка остальных Python-зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Указываем порт
EXPOSE 8501

# Указание команды для запуска
CMD ["streamlit", "run", "script.py", "--server.port=8501", "--server.address=0.0.0.0"]

