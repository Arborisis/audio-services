from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.config import settings
import logging

bearer_scheme = HTTPBearer(auto_error=False)
logger = logging.getLogger(__name__)


def verify_token(credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)) -> str:
    if not credentials:
        raise HTTPException(status_code=401, detail="Authorization header missing.")

    logger.info(f"Token received length: {len(credentials.credentials)}, prefix: {credentials.credentials[:10]}")
    logger.info(f"Expected token length: {len(settings.analyzer_secret)}, prefix: {settings.analyzer_secret[:10]}")
    logger.info(f"Tokens match: {credentials.credentials == settings.analyzer_secret}")

    if credentials.credentials != settings.analyzer_secret:
        raise HTTPException(status_code=401, detail="Invalid token.")

    return credentials.credentials
