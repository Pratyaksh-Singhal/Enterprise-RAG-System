import time
import logfire
import torch
from sentence_transformers import SentenceTransformer
from app.config import settings

BATCH_SIZE = 32

_active_model = None
_model_type: str | None = None  # "primary" or "fallback"

# Force CPU — the installed PyTorch wheels don't ship kernels for sm_120 (RTX 5050 Blackwell).
# CPU is fast enough for single-query embedding at inference time.
_device = "cpu"

# ── Model initialisation ───────────────────────────────────────────────────────

def _load_primary():
    """Try to load the primary model."""
    try:
        logfire.info(f"Loading primary embedding model (mxbai-embed-large-v1) on device: {_device.upper()}.")
        model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", device=_device)
        return model
    except Exception as e:
        logfire.warning(f"Primary model load failed: {e}. Will use fallback.")
        return None

def _load_fallback():
    logfire.info("Loading sentence-transformers fallback (sentence-transformers/all-mpnet-base-v2).")
    return SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


def _init():
    """Initialise embedding model once per process. Called lazily on first use."""
    global _active_model, _model_type
    if _active_model is not None:
        return

    primary = _load_primary()
    if primary:
        _active_model = primary
        _model_type = "primary"
    else:
        _active_model = _load_fallback()
        _model_type = "fallback"


# ── Public helpers ─────────────────────────────────────────────────────────────

def get_embedding_dim() -> int:
    """Return the vector dimension for the active model. Call after _init()."""
    _init()
    if hasattr(_active_model, "get_sentence_embedding_dimension"):
        return _active_model.get_sentence_embedding_dimension()
    return _active_model.get_embedding_dimension()


# ── Batch embedding with retry ─────────────────────────────────────────────────

def _embed_batch(batch: list[str]) -> list[list[float]]:
    # Both use SentenceTransformers
    return _active_model.encode(batch, show_progress_bar=False).tolist()


# ── Public API (same signatures as before) ─────────────────────────────────────

def embed_query(query: str) -> list[float]:
    _init()
    if _model_type == "primary":
        # mxbai-embed-large-v1 requires prefix for query encoding
        formatted = f"Represent this sentence for searching relevant passages: {query}"
        return _active_model.encode([formatted])[0].tolist()
    return _active_model.encode([query])[0].tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    _init()
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        with logfire.span("Embed batch", model=_model_type, start=i, size=len(batch)):
            all_embeddings.extend(_embed_batch(batch))
    return all_embeddings