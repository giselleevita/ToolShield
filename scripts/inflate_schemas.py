#!/usr/bin/env python3
"""Inflate tool schemas to simulate enterprise-length API definitions.

Reads an existing dataset.jsonl and produces a new JSONL file where each
record's tool_schema has been deterministically extended with realistic
enterprise properties (audit fields, pagination, RBAC, etc.) until the
serialised JSON reaches a target character length.

The inflation is:
- Deterministic (seeded PRNG per record)
- Content-preserving (original properties kept verbatim)
- Realistic (added properties mirror real enterprise API patterns)

Usage:
    python scripts/inflate_schemas.py
    python scripts/inflate_schemas.py --target-chars 4000
    python scripts/inflate_schemas.py --input data/dataset.jsonl --output data/dataset_longschema.jsonl
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# ── Enterprise property templates ────────────────────────────────────────────
# These mirror real enterprise API patterns (Salesforce, ServiceNow, SAP, etc.)

ENTERPRISE_PROPERTIES = [
    ("auditTrailId", "string", "UUID for compliance audit trail tracking and SOX reporting"),
    ("correlationId", "string", "Distributed tracing correlation ID for cross-service request tracking"),
    ("requestedBy", "string", "Principal identity (email or service account) initiating the request"),
    ("approvedBy", "string", "Manager or security officer who approved this operation, required for privileged actions"),
    ("departmentCode", "string", "Organizational unit code for cost allocation and access control (e.g., 'ENG-042')"),
    ("costCenter", "string", "Financial cost center code for billing and chargeback purposes"),
    ("environmentId", "string", "Deployment environment identifier: production, staging, development, sandbox"),
    ("tenantId", "string", "Multi-tenant isolation identifier for SaaS deployments"),
    ("regionCode", "string", "Geographic region code for data residency compliance (e.g., 'eu-west-1', 'us-east-2')"),
    ("dataClassification", "string", "Data sensitivity label: public, internal, confidential, restricted, top-secret"),
    ("retentionPolicyId", "string", "Data retention policy identifier for GDPR/CCPA compliance"),
    ("encryptionKeyId", "string", "KMS encryption key identifier for data-at-rest encryption"),
    ("accessLevel", "string", "Required RBAC access level: viewer, editor, admin, superadmin, owner"),
    ("mfaVerified", "boolean", "Whether multi-factor authentication was completed for this request"),
    ("sessionToken", "string", "Ephemeral session token from identity provider (JWT or opaque)"),
    ("idempotencyKey", "string", "Client-generated idempotency key for safe retry of mutating operations"),
    ("rateLimitGroup", "string", "Rate limiting group identifier for throttling and quota management"),
    ("priorityLevel", "integer", "Request priority for queue ordering: 0 (critical) to 9 (background)"),
    ("callbackUrl", "string", "Webhook URL for asynchronous operation completion notification"),
    ("timeoutSeconds", "integer", "Maximum execution timeout in seconds before automatic cancellation"),
    ("paginationToken", "string", "Opaque cursor token for resuming paginated result sets"),
    ("pageSize", "integer", "Number of results per page for paginated queries (max 1000)"),
    ("sortField", "string", "Field name to sort results by, must be an indexed column"),
    ("sortOrder", "string", "Sort direction: ascending or descending, defaults to ascending"),
    ("filterExpression", "string", "OData-style filter expression for server-side result filtering"),
    ("includeDeleted", "boolean", "Whether to include soft-deleted records in the response"),
    ("expandRelations", "string", "Comma-separated list of related entities to eagerly load"),
    ("fieldsProjection", "string", "Comma-separated list of fields to include in response (sparse fieldset)"),
    ("locale", "string", "BCP-47 locale tag for response localisation (e.g., 'en-US', 'de-DE')"),
    ("timezone", "string", "IANA timezone identifier for date/time interpretation (e.g., 'Europe/Berlin')"),
    ("dryRun", "boolean", "If true, validate the request without executing side effects"),
    ("webhookSecret", "string", "HMAC secret for signing webhook payloads to prevent tampering"),
    ("complianceFlags", "string", "Comma-separated compliance framework tags: SOC2, HIPAA, PCI-DSS, ISO27001"),
    ("changeTicketId", "string", "ITSM change management ticket reference for production modifications"),
    ("rollbackEnabled", "boolean", "Whether automatic rollback is enabled if operation fails validation"),
    ("batchId", "string", "Batch processing group identifier for bulk operations"),
    ("parentOperationId", "string", "Reference to parent operation for hierarchical audit trails"),
    ("notificationChannels", "string", "Comma-separated notification channels: email, slack, pagerduty, teams"),
    ("customMetadata", "object", "Free-form key-value metadata for customer-specific extensions and integrations"),
    ("sourceSystem", "string", "Originating system identifier for cross-platform data lineage tracking"),
    ("targetSystem", "string", "Destination system identifier for data synchronisation operations"),
    ("schemaVersion", "string", "API schema version for backward compatibility negotiation (semver)"),
    ("featureFlags", "string", "Comma-separated feature flag identifiers controlling experimental behaviour"),
    ("quotaProject", "string", "Project identifier for resource quota tracking and enforcement"),
    ("billingAccount", "string", "Billing account reference for usage-based pricing and invoicing"),
    ("serviceLevelObjective", "string", "SLO identifier for performance monitoring and alerting thresholds"),
    ("retryPolicy", "string", "Retry policy identifier: exponential-backoff, linear, none"),
    ("maxRetries", "integer", "Maximum number of automatic retry attempts before failure"),
    ("circuitBreakerId", "string", "Circuit breaker configuration identifier for fault tolerance"),
    ("cacheTtlSeconds", "integer", "Time-to-live for response caching in seconds, 0 disables caching"),
    ("etagVersion", "string", "Entity tag for optimistic concurrency control and conditional requests"),
    ("acceptEncoding", "string", "Preferred response encoding: gzip, br, zstd, identity"),
]


def inflate_schema(
    original_schema: dict,
    target_chars: int,
    seed: int,
) -> dict:
    """Deterministically inflate a JSON schema to target character length.

    Adds enterprise-realistic properties from the template pool until
    the serialised JSON reaches target_chars.

    Args:
        original_schema: The original tool JSON schema.
        target_chars: Target character length for serialised schema.
        seed: Random seed for deterministic property selection.

    Returns:
        Inflated copy of the schema.
    """
    schema = copy.deepcopy(original_schema)
    rng = random.Random(seed)

    # Ensure "properties" dict exists
    if "properties" not in schema:
        schema["properties"] = {}

    # Shuffle enterprise properties deterministically
    pool = list(ENTERPRISE_PROPERTIES)
    rng.shuffle(pool)

    idx = 0
    while len(json.dumps(schema, separators=(",", ":"))) < target_chars:
        if idx >= len(pool):
            # Exhausted pool — add numbered variants
            base_name, base_type, base_desc = pool[idx % len(pool)]
            prop_name = f"{base_name}_{idx // len(pool) + 1}"
            prop_type = base_type
            prop_desc = f"{base_desc} (extended field {idx // len(pool) + 1})"
        else:
            prop_name, prop_type, prop_desc = pool[idx]

        schema["properties"][prop_name] = {
            "type": prop_type,
            "description": prop_desc,
        }
        idx += 1

    return schema


def main() -> None:
    parser = argparse.ArgumentParser(description="Inflate tool schemas to enterprise length")
    parser.add_argument("--input", type=Path,
                        default=PROJECT_ROOT / "data" / "dataset.jsonl")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "data" / "dataset_longschema.jsonl")
    parser.add_argument("--target-chars", type=int, default=4000,
                        help="Target schema JSON length in characters")
    parser.add_argument("--seed", type=int, default=1337,
                        help="Base seed for deterministic inflation")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        print("Run 'make data' first to generate the base dataset.")
        raise SystemExit(1)

    print(f"Inflating schemas to ~{args.target_chars} chars")
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    schema_lengths = []
    with open(args.input) as fin, open(args.output, "w") as fout:
        for line_no, line in enumerate(fin):
            record = json.loads(line)

            # Deterministic per-record seed: base_seed + line_number
            record_seed = args.seed + line_no

            record["tool_schema"] = inflate_schema(
                record["tool_schema"],
                target_chars=args.target_chars,
                seed=record_seed,
            )

            schema_str = json.dumps(record["tool_schema"], separators=(",", ":"))
            schema_lengths.append(len(schema_str))

            fout.write(json.dumps(record) + "\n")
            count += 1

    import statistics
    print(f"\nInflated {count} records")
    print(f"Schema length: min={min(schema_lengths)}, "
          f"median={statistics.median(schema_lengths):.0f}, "
          f"max={max(schema_lengths)}, "
          f"mean={statistics.mean(schema_lengths):.0f}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
