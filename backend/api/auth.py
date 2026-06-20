from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.core.security import hash_password, verify_password, create_access_token
from backend.schemas.auth import RegisterRequest, LoginRequest, TokenResponse
from backend.core.postgre import get_db

from backend.api.deps import get_current_user

router = APIRouter(prefix="/auth", tags=["Auth"])


@router.post("/register", response_model=TokenResponse)
def register(data: RegisterRequest, db: Session = Depends(get_db)):
    existing_user = db.execute(
        text("""
            SELECT id
            FROM users
            WHERE email = :email
        """),
        {"email": data.email}
    ).first()

    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email already exists"
        )

    try:
        tenant = db.execute(
            text("""
                INSERT INTO tenants (name)
                VALUES (:name)
                RETURNING id
            """),
            {"name": data.tenant_name}
        ).first()

        tenant_id = tenant.id

        hashed_password = hash_password(data.password)

        user = db.execute(
            text("""
                INSERT INTO users (
                    tenant_id,
                    email,
                    password_hash,
                    full_name,
                    role,
                    is_active
                )
                VALUES (
                    :tenant_id,
                    :email,
                    :password_hash,
                    :full_name,
                    :role,
                    :is_active
                )
                RETURNING id, tenant_id, email, role
            """),
            {
                "tenant_id": tenant_id,
                "email": data.email,
                "password_hash": hashed_password,
                "full_name": data.full_name,
                "role": "admin",
                "is_active": True,
            }
        ).first()

        db.commit()

    except Exception:
        db.rollback()
        raise

    access_token = create_access_token(
    user_id=user.id,
    tenant_id=user.tenant_id,
    role=user.role,
)

    return TokenResponse(
        access_token=access_token,
        refresh_token=None,
        token_type="bearer"
    )


@router.post("/login", response_model=TokenResponse)
def login(data: LoginRequest, db: Session = Depends(get_db)):
    user = db.execute(
        text("""
            SELECT id, tenant_id, email, password_hash, role, is_active
            FROM users
            WHERE email = :email
        """),
        {"email": data.email}
    ).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )

    if not verify_password(data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    access_token = create_access_token(
    user_id=user.id,
    tenant_id=user.tenant_id,
    role=user.role,
)

    return TokenResponse(
        access_token=access_token,
        refresh_token=None,
        token_type="bearer"
    )



@router.get("/me")
def get_me(current_user: dict = Depends(get_current_user)):
    return current_user