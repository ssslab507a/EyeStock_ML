FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

WORKDIR /llm1

# 빌드 필수 도구
#RUN apt-get update && apt-get install -y build-essential git cmake && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git cmake \
    libxml2-dev libxslt1-dev libjpeg-dev zlib1g-dev \
 && rm -rf /var/lib/apt/lists/*

# # 라이브러리 설치
# COPY requirements.txt /llm1/requirements.txt
# RUN pip install -r requirements.txt
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir \
    fastapi uvicorn[standard] \
    transformers==4.45.2 huggingface_hub accelerate \
    sentence-transformers \
    langchain==0.2.16 \
    langchain-community==0.2.16 \
    langchain-text-splitters==0.2.4 \
    langchain-chroma==0.1.4 \
    python-dotenv newspaper3k \
    lxml_html_clean>=0.4.1

# 기본 진입점: bash로 들어가도록
CMD ["/bin/bash"]


