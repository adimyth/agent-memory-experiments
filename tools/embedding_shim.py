# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["fastapi", "uvicorn", "sentence-transformers"]
# ///

"""An OpenAI-compatible /v1/embeddings endpoint backed by the local bge-m3.

Letta's server takes an `embedding_config` with a custom `embedding_endpoint`, so
pointing that at this shim puts Letta on exactly the same embedding weights as every
other system in this repo while its chat model still goes to the real OpenAI API.

The alternative was running Ollama, which means a 4GB image plus another copy of the
model in Ollama's own format. This reuses the weights already in the HuggingFace
cache and is the same model, not just the same model name.

    uv run --script tools/embedding_shim.py

Listens on 8399. Reachable from a container as http://host.docker.internal:8399/v1.
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

MODEL_NAME = os.environ.get("EXPERIMENT_EMBEDDER", "BAAI/bge-m3")

app = FastAPI()
model = SentenceTransformer(MODEL_NAME)


class EmbeddingRequest(BaseModel):
    input: str | list[str]
    model: str | None = None


@app.post("/v1/embeddings")
def embeddings(req: EmbeddingRequest) -> dict:
    texts = [req.input] if isinstance(req.input, str) else list(req.input)
    vectors = model.encode(texts, normalize_embeddings=True).tolist()
    return {
        "object": "list",
        "model": req.model or MODEL_NAME,
        "data": [
            {"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vectors)
        ],
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_NAME}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("AMX_SHIM_PORT", "8399")))
