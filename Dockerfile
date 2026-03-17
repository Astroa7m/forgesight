FROM python:3.13-slim

RUN apt-get update && apt-get install -y \
    libglib2.0-0t64 \
    libgl1 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# so it gets its own cache layer cuz its the largest lib
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# install paddlepaddle from official source
RUN pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

WORKDIR /app

# applying pip cache for faster builds
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# copy application code
COPY app.py .
COPY solution.py .
COPY extractors.py .
COPY detection_helpers.py .
COPY regex_helper.py .

# copy trained model
COPY forgery_model.pkl .

EXPOSE 8501
ENV GROQ_API_KEY=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
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]