"""FastAPI guard service for prompt injection detection.

This module provides a REST API for real-time prompt injection detection
with support for FPR budget-based thresholds.

Usage:
    uvicorn toolshield.demo.app:app --host 0.0.0.0 --port 8000
    
    # Or via the run script:
    python scripts/run_demo.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_MODEL_PATH = os.getenv("TOOLSHIELD_MODEL_PATH", "outputs/tfidf_lr/")
DEFAULT_AUDIT_LOG = os.getenv("TOOLSHIELD_AUDIT_LOG", "data/audit/guard_audit.jsonl")

# FPR budget thresholds (set via config or environment)
DEFAULT_THRESHOLDS = {
    0.01: 0.7,   # Conservative: FPR budget 1%
    0.03: 0.5,   # Moderate: FPR budget 3%
    0.05: 0.4,   # Permissive: FPR budget 5%
}


class GuardRequest(BaseModel):
    """Request model for the /guard endpoint.
    
    Attributes:
        prompt: The prompt text to classify.
        tool_name: Name of the target tool (optional).
        tool_schema: JSON schema of the tool parameters (optional).
        tool_description: Natural language description of the tool (optional).
        role_sequence: Conversation role sequence (optional).
        fpr_budget: FPR budget for threshold selection (0.01, 0.03, or 0.05).
    """
    
    prompt: str = Field(..., description="The prompt text to classify")
    tool_name: str | None = Field(None, description="Target tool name")
    tool_schema: dict[str, Any] | None = Field(None, description="Tool JSON schema")
    tool_description: str | None = Field(None, description="Tool description")
    role_sequence: list[str] | None = Field(
        default=["system", "user"],
        description="Conversation role sequence"
    )
    fpr_budget: float = Field(
        default=0.03,
        description="FPR budget (0.01, 0.03, or 0.05)"
    )


class GuardResponse(BaseModel):
    """Response model for the /guard endpoint.
    
    Attributes:
        decision: The guard decision (ALLOW or BLOCK).
        score: The raw prediction score (0-1, higher = more likely attack).
        threshold: The threshold used for the decision.
        audit_id: Unique identifier for this request.
        explanation: Brief explanation of the decision.
        latency_ms: Processing time in milliseconds.
    """
    
    decision: Literal["ALLOW", "BLOCK"]
    score: float
    threshold: float
    audit_id: str
    explanation: str
    latency_ms: float


class AuditEntry(BaseModel):
    """Audit log entry for tracking guard decisions.
    
    Note: Does NOT store raw prompt text, only a hash for privacy.
    """
    
    audit_id: str
    timestamp: str
    prompt_hash: str
    tool_name: str | None
    role_sequence: list[str] | None
    fpr_budget: float
    score: float
    threshold: float
    decision: str
    latency_ms: float


# Global model cache
_model_cache: dict[str, Any] = {}
_thresholds: dict[float, float] = DEFAULT_THRESHOLDS.copy()


def _hash_prompt(prompt: str) -> str:
    """Create a SHA256 hash of the prompt for audit logging.
    
    Args:
        prompt: The raw prompt text.
        
    Returns:
        First 32 characters of the SHA256 hash.
    """
    return hashlib.sha256(prompt.encode()).hexdigest()[:32]


def _load_model(model_path: str) -> Any:
    """Load a model from disk, with caching.
    
    Args:
        model_path: Path to the model directory.
        
    Returns:
        Loaded classifier instance.
    """
    if model_path in _model_cache:
        return _model_cache[model_path]
    
    model_dir = Path(model_path)
    config_path = model_dir / "config.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found: {config_path}")
    
    with config_path.open() as f:
        model_config = json.load(f)
    
    model_type = model_config.get("model_type", "unknown")
    logger.info(f"Loading {model_type} model from {model_path}")
    
    if model_type == "heuristic":
        from toolshield.models.heuristic import HeuristicClassifier
        model = HeuristicClassifier.load(model_dir)
    elif model_type == "heuristic_score":
        from toolshield.models.heuristic_score import ScoredHeuristicClassifier
        model = ScoredHeuristicClassifier.load(model_dir)
    elif model_type == "tfidf_lr":
        from toolshield.models.tfidf_lr import TfidfLRClassifier
        model = TfidfLRClassifier.load(model_dir)
    elif model_type == "transformer":
        from toolshield.models.transformer import TransformerClassifier
        model = TransformerClassifier.load(model_dir)
    elif model_type == "context_transformer":
        from toolshield.models.context_transformer import ContextTransformerClassifier
        model = ContextTransformerClassifier.load(model_dir)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    _model_cache[model_path] = model
    logger.info(f"Model loaded successfully")
    return model


def _write_audit_entry(entry: AuditEntry, audit_path: str) -> None:
    """Write an audit entry to the JSONL log file.
    
    Args:
        entry: The audit entry to write.
        audit_path: Path to the audit log file.
    """
    audit_file = Path(audit_path)
    audit_file.parent.mkdir(parents=True, exist_ok=True)
    
    with audit_file.open("a") as f:
        f.write(entry.model_dump_json() + "\n")


def _get_threshold(fpr_budget: float) -> float:
    """Get the threshold for a given FPR budget.
    
    Args:
        fpr_budget: The FPR budget (0.01, 0.03, or 0.05).
        
    Returns:
        The corresponding threshold.
    """
    # Find closest budget
    available_budgets = sorted(_thresholds.keys())
    closest_budget = min(available_budgets, key=lambda x: abs(x - fpr_budget))
    return _thresholds[closest_budget]


def _generate_explanation(score: float, threshold: float, decision: str) -> str:
    """Generate a brief explanation for the guard decision.
    
    Args:
        score: The prediction score.
        threshold: The threshold used.
        decision: The decision made (ALLOW or BLOCK).
        
    Returns:
        Brief explanation string.
    """
    if decision == "BLOCK":
        if score > 0.9:
            return "High confidence: prompt exhibits strong injection patterns"
        elif score > 0.7:
            return "Moderate confidence: prompt contains suspicious patterns"
        else:
            return "Threshold exceeded: prompt flagged as potential injection"
    else:
        if score < 0.1:
            return "High confidence: prompt appears benign"
        elif score < 0.3:
            return "Moderate confidence: prompt appears safe"
        else:
            return "Below threshold: prompt allowed but monitor recommended"


# Initialize FastAPI app
app = FastAPI(
    title="ToolShield Guard API",
    description="Prompt injection detection for tool-using LLM agents",
    version="0.1.0",
)


@app.on_event("startup")
async def startup_event() -> None:
    """Pre-load model on startup and warmup if applicable."""
    try:
        model = _load_model(DEFAULT_MODEL_PATH)
        logger.info(f"Model loaded: {DEFAULT_MODEL_PATH}")
        
        # Warmup transformer models for fast inference
        if hasattr(model, 'warmup') and callable(getattr(model, 'warmup')):
            logger.info("Warming up transformer model...")
            model.warmup(n_samples=20)
            logger.info("Model warmup complete")
    except Exception as e:
        logger.warning(f"Could not pre-load model: {e}")
        logger.warning("Model will be loaded on first request")


@app.get("/")
async def root() -> dict[str, Any]:
    """Root endpoint with API info."""
    return {
        "service": "ToolShield Guard API",
        "version": "0.1.0",
        "endpoints": {
            "/guard": "POST - Evaluate a prompt for injection",
            "/health": "GET - Health check",
        }
    }


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    model_loaded = DEFAULT_MODEL_PATH in _model_cache
    return {
        "status": "healthy",
        "model_loaded": str(model_loaded),
        "model_path": DEFAULT_MODEL_PATH,
    }


@app.post("/guard", response_model=GuardResponse)
async def guard(request: GuardRequest) -> GuardResponse:
    """Evaluate a prompt for potential injection attacks.
    
    Args:
        request: The guard request containing prompt and context.
        
    Returns:
        GuardResponse with decision, score, and audit information.
        
    Raises:
        HTTPException: If model loading fails or processing error occurs.
    """
    import time
    start_time = time.perf_counter()
    
    # Generate audit ID
    audit_id = str(uuid.uuid4())[:8]
    
    try:
        # Load model
        model = _load_model(DEFAULT_MODEL_PATH)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=503, detail=f"Model not available: {e}")
    
    # Get threshold for budget
    threshold = _get_threshold(request.fpr_budget)
    
    # Create a record for prediction
    from toolshield.data.schema import DatasetRecord
    
    record = DatasetRecord(
        id=audit_id,
        language="en",
        role_sequence=request.role_sequence or ["system", "user"],
        tool_name=request.tool_name or "unknown",
        tool_schema=request.tool_schema or {},
        tool_description=request.tool_description or "",
        prompt=request.prompt,
        label_binary=0,  # Unknown, we're predicting
        attack_family=None,
        attack_goal=None,
        template_id="runtime",
        variant_id="v0",
        seed=0,
    )
    
    # Predict
    scores = model.predict_scores([record])
    score = float(scores[0])
    
    # Make decision
    decision: Literal["ALLOW", "BLOCK"] = "BLOCK" if score >= threshold else "ALLOW"
    
    # Calculate latency
    latency_ms = (time.perf_counter() - start_time) * 1000
    
    # Generate explanation
    explanation = _generate_explanation(score, threshold, decision)
    
    # Write audit log (without raw prompt)
    audit_entry = AuditEntry(
        audit_id=audit_id,
        timestamp=datetime.utcnow().isoformat(),
        prompt_hash=_hash_prompt(request.prompt),
        tool_name=request.tool_name,
        role_sequence=request.role_sequence,
        fpr_budget=request.fpr_budget,
        score=score,
        threshold=threshold,
        decision=decision,
        latency_ms=latency_ms,
    )
    
    try:
        _write_audit_entry(audit_entry, DEFAULT_AUDIT_LOG)
    except Exception as e:
        logger.warning(f"Failed to write audit log: {e}")
    
    return GuardResponse(
        decision=decision,
        score=round(score, 4),
        threshold=round(threshold, 4),
        audit_id=audit_id,
        explanation=explanation,
        latency_ms=round(latency_ms, 2),
    )


@app.post("/configure")
async def configure(
    thresholds: dict[float, float] | None = None,
    model_path: str | None = None,
) -> dict[str, Any]:
    """Configure the guard service (admin endpoint).
    
    Args:
        thresholds: New threshold mapping {fpr_budget: threshold}.
        model_path: New model path to load.
        
    Returns:
        Current configuration.
    """
    global _thresholds
    
    if thresholds:
        _thresholds.update(thresholds)
        logger.info(f"Updated thresholds: {_thresholds}")
    
    if model_path:
        try:
            _load_model(model_path)
            logger.info(f"Loaded new model from: {model_path}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load model: {e}")
    
    return {
        "thresholds": _thresholds,
        "model_path": DEFAULT_MODEL_PATH,
        "cached_models": list(_model_cache.keys()),
    }
