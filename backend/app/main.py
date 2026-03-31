from fastapi import FastAPI
from app.routers import listings, predict, chatbot

app = FastAPI(title="PRISM - Paris Real Estate ML API")

app.include_router(listings.router, prefix="/api")
app.include_router(predict.router, prefix="/api")
app.include_router(chatbot.router, prefix="/api")
