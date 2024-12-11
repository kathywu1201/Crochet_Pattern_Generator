from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
# from api.routers import newsletter, podcast
from api.routers import llm_rag_chat, llm_chat, llm_llama_chat
from fastapi.routing import APIRoute

# Setup FastAPI app
app = FastAPI(title="API Server", description="API Server", version="v1")

# Enable CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=False,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
@app.get("/")
async def get_index():
    return {"message": "The API is working!!!"}

# Additional routers here
# app.include_router(newsletter.router, prefix="/newsletters")
# app.include_router(podcast.router, prefix="/podcasts")
app.include_router(llm_chat.router, prefix="/llm")
app.include_router(llm_llama_chat.router, prefix="/llm-llama")
app.include_router(llm_rag_chat.router, prefix="/llm-rag")
# app.include_router(llm_agent_chat.router, prefix="/llm-agent")

