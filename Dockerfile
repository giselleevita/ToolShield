FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml requirements.txt ./
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY tests ./tests
COPY data ./data
COPY outputs ./outputs

RUN python -m pip install --upgrade pip && pip install -e .

EXPOSE 8000
CMD ["uvicorn", "toolshield.demo.app:app", "--host", "0.0.0.0", "--port", "8000"]
