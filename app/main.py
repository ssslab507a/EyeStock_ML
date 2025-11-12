from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from app.routers import health, chat, news
from app.models.chat import ChatRequest, ChatResponse
from app.services.rag_service import process_chat

app = FastAPI(title="EyeStock Chat API", version="1.0.0", docs_url="/docs",openapi_url="/openapi.json" )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(chat.router)
app.include_router(news.router)