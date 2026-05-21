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

# Vérifier les imports légers uniquement (les libs lourdes sont lazy-load)
echo "Vérification des dépendances..."
python -c "import fastapi; print('✓ FastAPI')" || echo "✗ FastAPI manquant"
python -c "import uvicorn; print('✓ Uvicorn')" || echo "✗ Uvicorn manquant"

echo ""
echo "Démarrage du serveur sur le port 8000..."
echo ""

# Démarrer uvicorn avec 1 seul worker pour économiser la mémoire
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 --log-level info
