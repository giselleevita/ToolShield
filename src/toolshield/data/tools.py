"""Tool definitions for the enterprise tool-using agent scenario.

This module defines the four enterprise tools used in the dataset:
- getCustomerRecord: Retrieve customer PII
- searchTickets: Search support tickets
- exportReport: Export data reports
- resetUserMFA: Reset user MFA (privileged operation)
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Tool(BaseModel):
    """Definition of an enterprise tool with schema and description.

    Attributes:
        name: Unique tool identifier.
        description: Natural language description of the tool.
        schema_: JSON schema defining the tool's parameters.
    """

    name: str = Field(..., description="Tool name/identifier")
    description: str = Field(..., description="Natural language description")
    schema_: dict[str, Any] = Field(..., alias="schema", description="JSON schema for parameters")

    class Config:
        """Pydantic configuration."""

        populate_by_name = True


# Tool: getCustomerRecord
GET_CUSTOMER_RECORD = Tool(
    name="getCustomerRecord",
    description=(
        "Retrieve a customer's record including personal information, account details, "
        "and contact information. Requires valid customer ID. Returns sensitive PII."
    ),
    schema={
        "type": "object",
        "properties": {
            "customerId": {
                "type": "string",
                "description": "The unique identifier for the customer",
            }
        },
        "required": ["customerId"],
    },
)

# Tool: searchTickets
SEARCH_TICKETS = Tool(
    name="searchTickets",
    description=(
        "Search support tickets by query and date range. Returns ticket summaries "
        "including customer issues, support agent responses, and resolution status."
    ),
    schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query for ticket content",
            },
            "dateRange": {
                "type": "string",
                "description": "Date range filter (e.g., 'last_7_days', 'last_30_days', '2024-01-01:2024-12-31')",
            },
        },
        "required": ["query", "dateRange"],
    },
)

# Tool: exportReport
EXPORT_REPORT = Tool(
    name="exportReport",
    description=(
        "Export a data report of the specified type and scope. Report types include "
        "sales, customer_activity, support_metrics, and financial_summary. "
        "Scope can be 'all', 'department', or specific team identifiers."
    ),
    schema={
        "type": "object",
        "properties": {
            "reportType": {
                "type": "string",
                "enum": ["sales", "customer_activity", "support_metrics", "financial_summary"],
                "description": "Type of report to generate",
            },
            "scope": {
                "type": "string",
                "description": "Scope of the report ('all', 'department', or team ID)",
            },
        },
        "required": ["reportType", "scope"],
    },
)

# Tool: resetUserMFA
RESET_USER_MFA = Tool(
    name="resetUserMFA",
    description=(
        "Reset multi-factor authentication for a user account. This is a privileged "
        "operation that should only be performed after verifying the user's identity "
        "through an approved verification process. Audit logged."
    ),
    schema={
        "type": "object",
        "properties": {
            "userId": {
                "type": "string",
                "description": "The unique identifier for the user whose MFA will be reset",
            }
        },
        "required": ["userId"],
    },
)

# All tools as a list for easy iteration
TOOLS: list[Tool] = [
    GET_CUSTOMER_RECORD,
    SEARCH_TICKETS,
    EXPORT_REPORT,
    RESET_USER_MFA,
]

# Tool lookup by name
TOOL_BY_NAME: dict[str, Tool] = {tool.name: tool for tool in TOOLS}

# Tool names as a list
TOOL_NAMES: list[str] = [tool.name for tool in TOOLS]


def get_tool(name: str) -> Tool:
    """Get a tool by name.

    Args:
        name: The tool name.

    Returns:
        The Tool object.

    Raises:
        KeyError: If the tool name is not found.
    """
    if name not in TOOL_BY_NAME:
        raise KeyError(f"Unknown tool: {name}. Available tools: {TOOL_NAMES}")
    return TOOL_BY_NAME[name]


def get_tool_schema(name: str) -> dict[str, Any]:
    """Get a tool's JSON schema by name.

    Args:
        name: The tool name.

    Returns:
        The tool's JSON schema.
    """
    return get_tool(name).schema_


def get_tool_description(name: str) -> str:
    """Get a tool's description by name.

    Args:
        name: The tool name.

    Returns:
        The tool's description.
    """
    return get_tool(name).description
