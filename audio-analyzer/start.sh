#!/bin/bash
# Script de démarrage avec diagnostic

echo "========================================"
echo "Démarrage Audio Analyzer"
echo "========================================"
echo "Date: $(date)"
echo "Python: $(python --version)"
echo "Working dir: $(pwd)"
echo "User: $(whoami)"
echo ""

# Vérifier les imports
echo "Vérification des dépendances..."
python -c "import fastapi; print('✓ FastAPI')" || echo "✗ FastAPI manquant"
python -c "import uvicorn; print('✓ Uvicorn')" || echo "✗ Uvicorn manquant"
python -c "import librosa; print('✓ Librosa')" || echo "✗ Librosa manquant"
python -c "import tensorflow; print('✓ TensorFlow')" || echo "✗ TensorFlow manquant"
python -c "import birdnet_analyzer; print('✓ BirdNET')" || echo "✗ BirdNET manquant"

echo ""
echo "Démarrage du serveur sur le port 8000..."
echo ""

# Démarrer uvicorn avec 1 seul worker pour économiser la mémoire
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 --log-level info
