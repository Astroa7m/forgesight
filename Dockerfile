FROM python:3.13-slim

# install system dependencies needed by opencv and paddleocr
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy application code
COPY app.py .
COPY solution.py .
COPY extractors.py .
COPY model_helpers.py .
COPY detection_helpers.py .
COPY regex_helper.py .

# copy trained model
COPY forgery_model.pkl .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]