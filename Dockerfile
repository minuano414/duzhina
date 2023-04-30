FROM python:3.9-slim-buster

RUN apt-get update  && apt-get install -y libgl1-mesa-glx gcc python3-dev libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ./models /app/models

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

CMD ["tail", "-f", "/dev/null"]