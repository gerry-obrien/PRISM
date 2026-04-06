# =============================================================================
# Frontend — Streamlit UI
# Interface utilisateur pour explorer les listings et le chatbot
# =============================================================================
FROM python:3.11-slim

WORKDIR /app

# Dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Installer les dépendances Python
# Streamlit + bcrypt pour l'auth + pandas pour la data
RUN pip install --no-cache-dir \
    streamlit \
    pandas>=2.2.0 \
    requests>=2.31.0 \
    bcrypt \
    plotly

# Copier le code frontend
COPY frontend/ ./frontend/

EXPOSE 8501

CMD ["streamlit", "run", "frontend/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
