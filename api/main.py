"""FastAPI backend for Healthcare Content Generator."""
import sys
from pathlib import Path

# Ensure project root is on path when running: python api/main.py or uvicorn api.main:app
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from config import get_settings
from src.content_generator import create_generator

app = FastAPI(
    title="Healthcare Content Generator API",
    description="GenAI content generation for healthcare professionals: Patient Summaries, clinical notes, education handouts.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_settings = get_settings()


def _get_generator():
    config = {
        "openai_api_key": _settings.openai_api_key,
        "openai_api_base": _settings.openai_api_base,
        "llm_model": _settings.llm_model,
        "embedding_model": _settings.embedding_model,
        "chroma_persist_dir": _settings.chroma_persist_dir,
    }
    return create_generator(config)


class GenerateRequest(BaseModel):
    topic: str
    content_type: str = "patient_summary"
    temperature: float = 0.3
    max_tokens: int = 2048
    top_k_context: int = 5


class GenerateResponse(BaseModel):
    content: str
    content_type: str
    topic: str


@app.get("/health")
def health():
    return {"status": "ok", "service": "healthcare-content-generator"}


@app.get("/")
def serve_frontend():
    """Serve the web frontend so the page is same-origin and fetch works."""
    index_path = ROOT / "web" / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="web/index.html not found")
    return FileResponse(index_path)
@app.get("/result")
def serve_result_page():
    """Serve the result page that shows the generated document."""
    result_path = ROOT / "web" / "result.html"
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="web/result.html not found")
    return FileResponse(result_path)


@app.post("/generate", response_model=GenerateResponse)
def generate_content(req: GenerateRequest):
    """Generate healthcare content from a topic."""
    if not _settings.openai_api_key:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is not configured. Set it in .env.",
        )
    try:
        gen = _get_generator()
        content = gen(
            topic=req.topic,
            content_type=req.content_type,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            top_k_context=req.top_k_context,
        )
        return GenerateResponse(
            content=content,
            content_type=req.content_type,
            topic=req.topic,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/content-types")
def content_types():
    """List supported content types."""
    from src import prompts
    return {"content_types": list(prompts.CONTENT_TYPES.keys())}
