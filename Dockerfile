# Serves the trained models. Build after `make all` so artifacts/ exists.
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY pyproject.toml .
COPY src ./src
RUN pip install --no-cache-dir --no-deps -e .
COPY artifacts/student.joblib artifacts/baseline.joblib ./artifacts/
COPY artifacts/distilbert ./artifacts/distilbert
ENV NEWSCLF_BACKEND=linear
EXPOSE 8000
CMD ["uvicorn", "newsclf.serve:app", "--host", "0.0.0.0", "--port", "8000"]
