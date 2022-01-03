# syntax=docker/dockerfile:1
FROM python:3.9-slim
WORKDIR /
COPY . /
RUN pip install -r requirements.txt
CMD ["python", "src/app.py"]