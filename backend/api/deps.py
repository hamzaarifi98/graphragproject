from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from sqlalchemy import text

from backend.core.postgre import get_db
from backend.core.security import decode_access_token


security = HTTPBearer()

def get_current_user (
    credentials : HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
    ):

        token = credentials.credentials

        payload = decode_access_token(token)

        if not payload:
            raise HTTPException(detail="Invalid token")
        
        user_id = payload.get('sub')

        if not user_id:
            raise HTTPException(detail="Invalid token")
        
        user = db.execute(
        text("""
            SELECT id, tenant_id, email, full_name, role, is_active
            FROM users
            WHERE id = :user_id
        """),
        {"user_id": user_id}
    ).fetchone()
        
        if not user:
             raise HTTPException(details="User not found")
        
        if not user.is_active:
             raise HTTPException(details='user is not active')
        
        return {
        "id": str(user.id),
        "tenant_id": str(user.tenant_id),
        "email": user.email,
        "full_name": user.full_name,
        "role": user.role,
    }
        
    