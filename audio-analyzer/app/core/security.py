from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.config import settings

bearer_scheme = HTTPBearer(auto_error=False)


def verify_token(credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)) -> str:
    if not credentials:
        raise HTTPException(status_code=401, detail="Authorization header missing.")

    if credentials.credentials != settings.analyzer_secret:
        raise HTTPException(status_code=401, detail="Invalid token.")

    return credentials.credentials
