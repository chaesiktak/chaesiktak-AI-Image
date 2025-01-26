# 베이스 이미지 설정
FROM python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 파일 복사
COPY requirements.txt requirements.txt
COPY seg_operator.py seg_operator.py
COPY chaesiktak_weights.onnx chaesiktak_weights.onnx
COPY chaesiktak_classes.txt chaesiktak_classes.txt
COPY templates/index.html templates/index.html

# 종속성 설치
RUN pip install --no-cache-dir -r requirements.txt

# 컨테이너 실행 시 서버 시작
CMD ["python", "seg_operator.py"]

# 컨테이너에서 접근할 포트 설정
EXPOSE 5000