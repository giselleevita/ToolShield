"""Data schema definitions using Pydantic models."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AttackFamily(str, Enum):
    """Attack family identifiers for the MVT (Minimum Viable Threat) model."""

    AF1 = "AF1"  # Instruction Override
    AF2 = "AF2"  # Data Exfiltration
    AF3 = "AF3"  # Tool Hijacking
    AF4 = "AF4"  # Indirect Injection


class AttackGoal(str, Enum):
    """Specific attack goals for ASR computation."""

    POLICY_BYPASS = "policy_bypass"
    DATA_EXFILTRATION = "data_exfiltration"
    TOOL_HIJACK = "tool_hijack"
    PRIVILEGE_MISUSE = "privilege_misuse"


# Goals that count as unsafe outcomes for ASR
UNSAFE_GOALS: set[str] = {
    AttackGoal.POLICY_BYPASS.value,
    AttackGoal.DATA_EXFILTRATION.value,
    AttackGoal.TOOL_HIJACK.value,
    AttackGoal.PRIVILEGE_MISUSE.value,
}


class DatasetRecord(BaseModel):
    """A single record in the prompt injection detection dataset.

    This schema captures all necessary information for training and
    evaluating prompt injection detection models, including provenance
    fields (template_id, variant_id, seed) critical for proper splitting.

    Attributes:
        id: Unique identifier, deterministically generated from seed.
        language: Language code (e.g., "en").
        role_sequence: List of conversation roles (e.g., ["system", "user"]).
        tool_name: Name of the target tool.
        tool_schema: JSON schema of the tool's parameters.
        tool_description: Natural language description of the tool.
        prompt: The actual prompt text to classify.
        label_binary: Binary label (0=benign, 1=attack).
        attack_family: Attack family identifier (AF1-AF4) or None for benign.
        attack_goal: Specific attack goal or None for benign.
        template_id: Template identifier for split stratification.
        variant_id: Variant identifier within the template.
        seed: Random seed used for generation.
    """

    id: str = Field(..., description="Unique record identifier")
    language: str = Field(default="en", description="Language code")
    role_sequence: list[str] = Field(
        ..., description="Conversation role sequence (e.g., ['system', 'user'])"
    )
    tool_name: str = Field(..., description="Target tool name")
    tool_schema: dict[str, Any] = Field(..., description="JSON schema of tool parameters")
    tool_description: str = Field(..., description="Natural language tool description")
    prompt: str = Field(..., description="The prompt text to classify")
    label_binary: int = Field(..., ge=0, le=1, description="0=benign, 1=attack")
    attack_family: str | None = Field(
        default=None, description="Attack family (AF1-AF4) or None for benign"
    )
    attack_goal: str | None = Field(
        default=None, description="Attack goal or None for benign"
    )
    template_id: str = Field(..., description="Template identifier for splitting")
    variant_id: str = Field(..., description="Variant within template")
    seed: int = Field(..., description="Generation seed")

    class Config:
        """Pydantic model configuration."""

        json_schema_extra = {
            "example": {
                "id": "rec_001_benign_getCustomerRecord",
                "language": "en",
                "role_sequence": ["system", "user"],
                "tool_name": "getCustomerRecord",
                "tool_schema": {
                    "type": "object",
                    "properties": {"customerId": {"type": "string"}},
                    "required": ["customerId"],
                },
                "tool_description": "Retrieve customer record by ID",
                "prompt": "Please look up the account details for customer ID 12345.",
                "label_binary": 0,
                "attack_family": None,
                "attack_goal": None,
                "template_id": "benign_customer_lookup_01",
                "variant_id": "v1",
                "seed": 1337,
            }
        }

    def is_attack(self) -> bool:
        """Check if this record represents an attack."""
        return self.label_binary == 1

    def is_benign(self) -> bool:
        """Check if this record represents a benign prompt."""
        return self.label_binary == 0

    def has_unsafe_goal(self) -> bool:
        """Check if this record has an unsafe attack goal for ASR computation."""
        return self.attack_goal is not None and self.attack_goal in UNSAFE_GOALS


class SplitManifest(BaseModel):
    """Manifest for a dataset split, containing counts and metadata.

    Attributes:
        protocol: Split protocol name (S_random, S_attack_holdout, S_tool_holdout).
        seed: Random seed used for splitting.
        split_name: Name of this split (train, val, test).
        total_records: Total number of records in this split.
        label_counts: Counts by label_binary (0, 1).
        family_counts: Counts by attack_family (including None for benign).
        tool_counts: Counts by tool_name.
        template_ids: List of unique template_ids in this split.
    """

    protocol: str = Field(..., description="Split protocol name")
    seed: int = Field(..., description="Split seed")
    split_name: str = Field(..., description="Split name (train/val/test)")
    total_records: int = Field(..., description="Total record count")
    label_counts: dict[str, int] = Field(..., description="Counts by label")
    family_counts: dict[str, int] = Field(..., description="Counts by attack family")
    tool_counts: dict[str, int] = Field(..., description="Counts by tool")
    template_ids: list[str] = Field(..., description="Template IDs in this split")


class DatasetManifest(BaseModel):
    """Manifest for the full generated dataset.

    Attributes:
        seed: Random seed used for generation.
        total_records: Total number of records.
        label_counts: Counts by label_binary.
        family_counts: Counts by attack_family.
        tool_counts: Counts by tool_name.
        template_counts: Number of unique templates per category.
    """

    seed: int = Field(..., description="Generation seed")
    total_records: int = Field(..., description="Total record count")
    label_counts: dict[str, int] = Field(..., description="Counts by label")
    family_counts: dict[str, int] = Field(..., description="Counts by attack family")
    tool_counts: dict[str, int] = Field(..., description="Counts by tool")
    template_counts: dict[str, int] = Field(..., description="Unique templates per category")
