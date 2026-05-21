# Arborisis Audio Analyzer

Service Python FastAPI pour l'analyse audio automatique du pipeline Arborisis.

## Architecture

Ce service reçoit des requêtes du Worker Cloudflare, télécharge le fichier audio depuis R2, exécute l'analyse, upload les résultats dans R2, et notifie Laravel via callback.

```
Worker Cloudflare → POST /analyze → FastAPI Background Task
    → Download R2 → FFmpeg / FFprobe → Librosa → BirdNET
    → Upload résultats R2 → Callback Laravel
```

## Stack

- **FastAPI** + Uvicorn
- **boto3** (R2/S3)
- **librosa** + **numpy** + **scipy**
- **matplotlib** + **Pillow** (spectrogramme WebP)
- **BirdNET Analyzer** (classification espèces)
- **FFmpeg** / **FFprobe**

## Développement local

```bash
cd services/audio-analyzer

# Copier la config
cp .env.example .env

# Lancer avec docker-compose
docker-compose up --build

# Tester
 curl http://localhost:8000/health
 curl -X POST http://localhost:8000/analyze \
   -H "Authorization: Bearer <ANALYZER_SECRET>" \
   -H "Content-Type: application/json" \
   -d '{"sound_id": 1, "original_r2_key": "sounds/original/1/test.wav", "force": false}'
```

## Variables d'environnement

| Variable | Description |
|----------|-------------|
| `ANALYZER_SECRET` | Token Bearer pour authentifier les requêtes |
| `LARAVEL_API_URL` | URL de base de l'API Laravel |
| `LARAVEL_API_SECRET` | Token Bearer pour le callback Laravel |
| `R2_ENDPOINT` | Endpoint S3-compatible R2 |
| `R2_ACCESS_KEY_ID` | Clé d'accès R2 |
| `R2_SECRET_ACCESS_KEY` | Secret R2 |
| `R2_BUCKET` | Nom du bucket (défaut: `arborisis`) |
| `MAX_FILE_SIZE_MB` | Taille max fichier audio (défaut: 500) |
| `MAX_DURATION_SECONDS` | Durée max audio (défaut: 600) |
| `BIRDNET_CONFIDENCE_THRESHOLD` | Seuil bas envoye a BirdNET pour collecter les candidats (defaut: 0.25) |
| `BIRDNET_PUBLISH_CONFIDENCE_THRESHOLD` | Seuil minimal pour publier une detection isolee (defaut: 0.45) |
| `BIRDNET_REPEATED_CONFIDENCE_THRESHOLD` | Seuil moyen pour conserver une espece repetee dans plusieurs segments (defaut: 0.32) |
| `BIRDNET_MAX_DETECTIONS` | Nombre maximal de segments BirdNET conserves apres filtrage (defaut: 80) |
| `LOG_LEVEL` | Niveau de log (défaut: INFO) |

## Endpoints

### GET /health

Vérifie la disponibilité des dépendances (FFmpeg, BirdNET, librosa).

### POST /analyze

Déclenche une analyse asynchrone.

**Headers:**
- `Authorization: Bearer <ANALYZER_SECRET>`
- `Content-Type: application/json`

**Body:**
```json
{
  "sound_id": 123,
  "original_r2_key": "sounds/original/123/recording.wav",
  "force": false
}
```

**Response (202 Accepted):**
```json
{
  "status": "accepted",
  "sound_id": 123,
  "analysis_id": "uuid-v4",
  "message": "Analysis started"
}
```

## Fichiers générés dans R2

Pour un `sound_id` donné :

- `sounds/preview/{sound_id}/preview.mp3`
- `sounds/analysis/{sound_id}/waveform.json`
- `sounds/analysis/{sound_id}/spectrogram.webp`
- `sounds/analysis/{sound_id}/features.json`
- `sounds/analysis/{sound_id}/birdnet.json`
- `sounds/analysis/{sound_id}/summary.json`

## Tests

```bash
pytest
```

## Déploiement

### Docker (VPS / Contabo)

```bash
docker build -t arborisis-audio-analyzer .
docker run -d --env-file .env -p 8000:8000 --name analyzer arborisis-audio-analyzer
```

### Multi-instance sur VPS Worker dédié (Recommandé en production)

Pour scaler horizontalement, utiliser la config dans `infrastructure/audio-analyzer-worker/` :

```bash
cd ../../infrastructure/audio-analyzer-worker
docker compose up -d --build
```

Cela déploie 3 instances derrière un load balancer Nginx (round-robin pondéré par connexions actives). Voir [`infrastructure/audio-analyzer-worker/README.md`](../../infrastructure/audio-analyzer-worker/README.md) pour le setup complet.

### Cloudflare Containers (si disponible)

1. Builder et push l'image vers un registre
2. Déployer via le dashboard Cloudflare Containers
3. Configurer les variables d'environnement

## Sécurité

- Vérification `Authorization: Bearer` obligatoire
- Validation stricte des clés R2 (pas de `../`)
- Nettoyage automatique des fichiers temporaires
- Pas de secrets dans les logs
