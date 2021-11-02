# set base image
FROM python:3.8.12-slim

# set the working directory in the container
WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY models ./models

COPY ["predict.py", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]