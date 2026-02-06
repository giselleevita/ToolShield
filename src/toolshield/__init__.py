"""ToolShield: Prompt Injection Detection for Tool-Using LLM Agents.

A research-grade library for detecting prompt injection attacks
in tool-using LLM agents, featuring dataset generation, multiple
split protocols, and various classifier baselines.
"""

__version__ = "0.1.0"
__author__ = "ToolShield Research Team"

from toolshield.data.schema import DatasetRecord

__all__ = ["DatasetRecord", "__version__"]
