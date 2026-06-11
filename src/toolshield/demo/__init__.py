"""FastAPI demo service for ToolShield prompt injection detection."""

from toolshield.demo.app import GuardRequest, GuardResponse, app

__all__ = ["app", "GuardRequest", "GuardResponse"]
