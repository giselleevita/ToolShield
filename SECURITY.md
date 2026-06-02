# Security Policy

## Supported Versions

| Version | Supported |
|---------|----------|
| 0.x     | ✅        |

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Report vulnerabilities privately by emailing: **quaryn@protonmail.com**

Include:
- A description of the vulnerability
- Steps to reproduce
- Potential impact
- Any suggested fix (optional)

You can expect an acknowledgement within **48 hours** and a resolution or status update within **7 days**.

## Scope

This repository is a **research-grade prototype** for prompt injection detection in tool-using LLM agents. It is not a production security appliance.

- **In scope:** vulnerabilities in the tooling itself (e.g. unsafe deserialization, dependency vulnerabilities, data leakage in generated datasets)
- **Out of scope:** limitations of the detection model's coverage — see the README's security scope section

## Responsible Disclosure

We follow responsible disclosure practices. Credit will be given to reporters in the changelog unless anonymity is requested.
