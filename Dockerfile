FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /usr/app

RUN apt update && apt install -y \
    python3.10 \
    python-is-python3 \
    pip \
    git \
    git-lfs \
    curl \
    ffmpeg \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /usr/app/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' /usr/local/lib/python3.10/dist-packages/basicsr/data/degradations.py

ENV ENHANCE_METHOD='gfpgan'
ENV BACKGROUND_ENHANCEMENT='True'

COPY . .

CMD ["uvicorn", "app:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "80", "--workers", "1"]
