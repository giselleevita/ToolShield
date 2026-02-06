"""FastAPI demo service for ToolShield prompt injection detection."""

from toolshield.demo.app import app, GuardRequest, GuardResponse

__all__ = ["app", "GuardRequest", "GuardResponse"]
