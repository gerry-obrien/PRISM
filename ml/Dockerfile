# =============================================================================
# ML Pipeline — Génération de données + Entraînement du modèle
# Ce container s'exécute une fois puis s'arrête.
# =============================================================================
FROM python:3.11-slim

WORKDIR /app

# Dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Installer les dépendances Python du ML
COPY ml/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code ML
COPY ml/ ./ml/

# Créer les dossiers de sortie
RUN mkdir -p data ml/artifacts ml/plots

# Lancer : 1) génération des données  2) entraînement du modèle
CMD ["sh", "-c", "python ml/generate_data.py && python ml/train.py"]
