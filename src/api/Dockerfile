FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu24.04

WORKDIR /app/src/api

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
	software-properties-common \
	build-essential \
	gcc \
	libasound2-dev \
	&& add-apt-repository ppa:deadsnakes/ppa \
	&& apt-get update && apt-get install -y --no-install-recommends \
	python3.13 \
	python3.13-venv \
	python3.13-dev \
	&& rm -rf /var/lib/apt/lists/*

RUN python3.13 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY src/api/requirements.txt ./requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel \
	&& python -m pip install --no-cache-dir -r requirements.txt

COPY src /app/src

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]