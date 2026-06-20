from pydantic import BaseModel, EmailStr


class UserCreate(BaseModel):
    name:str
    username:str
    email: EmailStr
    password: str
    role: str


class UserLogin(BaseModel):
    email: EmailStr
    password:str







