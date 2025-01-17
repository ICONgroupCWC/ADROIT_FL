FROM python:3.10

WORKDIR /app

COPY requirements.txt .

ENV TORCH_CUDA_ARCH_LIST="None"
ENV CUDA_VISIBLE_DEVICES=""
ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir -r requirements.txt

COPY . .



ENTRYPOINT ["python3", "client_service.py"]
CMD []
