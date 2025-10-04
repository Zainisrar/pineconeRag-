from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import (
    users, rag
)
app = FastAPI(title="Neodustria API")
# :small_blue_diamond: Enable CORS so React (localhost:3000) can call the API
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    # add your production domains here when you deploy
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],   # GET, POST, PUT, OPTIONS etc.
    allow_headers=["*"],   # Accept, Authorization etc.
)
# Register routers
app.include_router(users.router)
app.include_router(rag.router)

@app.get("/")
def root():
    return {"message": "Neodustria API is running!"}





