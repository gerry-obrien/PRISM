"""
Chat router — single endpoint for the investment advisor chatbot.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
from app.services.chatbot import chat

router = APIRouter()


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = []


class ChatResponse(BaseModel):
    reply: str


@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """Send a message to the investment advisor and get a response."""
    # Convert history to list of dicts for the chatbot service
    history = [{"role": m.role, "content": m.content} for m in request.history]

    reply = chat(request.message, history)
    return ChatResponse(reply=reply)
