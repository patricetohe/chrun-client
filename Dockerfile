FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Dépendance OpenMP requise par XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Installer les dépendances en amont pour profiter du cache Docker
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code de l'application
COPY . .
# COPY src/serving/model/ /app/src/serving/model

# COPY src/serving/model/d94c01373d8c4a299552fb00e9331004/artifacts/model.pkl /app/model/
# COPY src/serving/model/d94c01373d8c4a299552fb00e9331004/artifacts/feature_columns.txt /app/model/



# Rendre src importable
ENV PYTHONPATH=/app/src

EXPOSE 8000

# Lancer l'API FastAPI
CMD ["python", "-m", "uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]


