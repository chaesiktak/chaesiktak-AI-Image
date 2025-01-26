FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt requirements.txt
COPY seg_operator.py seg_operator.py
COPY chaesiktak_weights.onnx chaesiktak_weights.onnx
COPY chaesiktak_classes.txt chaesiktak_classes.txt
COPY templates/index.html templates/index.html

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "seg_operator.py"]

EXPOSE 5000