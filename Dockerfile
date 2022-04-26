# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

WORKDIR /app

RUN pip3 install numpy
RUN pip3 install coverage

COPY . .

CMD ["bash", "script.sh"]
