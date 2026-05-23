# python image
FROM python:3.11-slim

# system-level environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TF_ENABLE_ONEDNN_OPTS=1

# set the working directory inside the container
WORKDIR /app

# install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install dependencies and clear pip cache
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# copy runtime application modules and structural configurations
COPY ./api /app/api
COPY ./src /app/src
COPY ./models /app/models

# include vocabulary/tokenization assets required by your model pipeline
COPY ./data /app/data
COPY ./outputs /app/outputs

# expose the internal port your FastAPI application runs on
EXPOSE 7860

# start api
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]