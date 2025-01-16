FROM python:3

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8200

ENTRYPOINT ["python3", "websocket_server.py"]
CMD ["--num_ues", "3"]