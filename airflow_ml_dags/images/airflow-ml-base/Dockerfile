FROM python:3.8-slim
RUN mkdir /build/
COPY requirements.txt /build/requirements.txt
COPY config.yaml /config.yaml
COPY entities.py /entities.py
COPY feature_transforms.py /feature_transforms.py

RUN pip install -r /build/requirements.txt
