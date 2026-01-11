FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV, TensorFlow, and any necessary libraries
RUN apt-get update && apt-get install -y \
    libatlas-base-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

COPY . /app
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Keep TensorFlow saving/loading weights in the legacy Keras 2 format that
# app.py's model_from_json() + load_weights() expects
ENV TF_USE_LEGACY_KERAS=1

CMD ["python", "app.py"]
