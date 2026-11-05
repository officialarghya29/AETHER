# Security — AETHER-ML

## 1. Threat model

| Asset | Threat | Control |
| --- | --- | --- |
| Uploaded images | Malicious payloads, decompression bombs | Byte and pixel limits before decode; Pillow format validation; generated storage names; content never executed |
| Review data | Unauthorised reads or writes | Optional bearer auth on review/feedback routes; no public bucket; paths never returned |
| API availability | Request floods, oversized batches | Sliding-window rate limit; batch cap of 16; request timeout |
| Database | SQL injection, credential leakage | SQLAlchemy parameterised queries; credentials from environment only |
| Model artifacts | Tampering or accidental replacement | Versioned checkpoints with metadata; promotion requires gate + approver; rollback recorded |
| Logs | Secret or payload leakage | Structured events log ids, actions and metrics, not file contents or tokens |

Out of scope for v1.0.0: multi-tenant isolation, per-user authorisation, network policy, WAF,
secrets-manager integration, and audit-log shipping to an external SIEM.

## 2. Input handling

- `aether.imaging.decode_image` enforces `upload_max_bytes` and `upload_max_pixels`, verifies the
  decoded format and rejects unsupported or corrupt data with `InvalidInputError`.
- Client filenames are ignored; stored files use `uuid4().hex` plus a neutral extension.
- Batch requests must share dimensions and stay within the file cap.
- SVG and other vector formats are not accepted.

## 3. Authentication and CORS

```yaml
api:
  auth:
    enabled: true
    token_env: AETHER_API_TOKEN
  cors_origins:
    - https://aether.example.com
  rate_limit_per_minute: 120
```

- Tokens are compared with `secrets.compare_digest`.
- Missing configuration fails closed: if auth is enabled and the token env var is empty, protected
  routes reject every request.
- CORS is an explicit allow-list; credentials are not enabled.
- Rate limiting is per client key with a sliding window; it is process-local and therefore a
  coarse protection, not a distributed quota.

## 4. Data handling

- Images are stored only for escalated, monitored, reviewed or abstained predictions, under the
  configured artifact root.
- The database stores relative references, never absolute paths or binary blobs.
- Review history is append-only; corrections add entries instead of rewriting them.
- Verified labels are required before a case can enter a candidate dataset.
- Deleting a prediction cascades to its evidence rows via foreign keys.

## 5. Container hardening

- Non-root user (`uid 10001`), no build toolchain in the runtime layer, distroless-adjacent slim base.
- No secrets baked into the image; compose passes them as environment variables.
- Health check uses an unauthenticated endpoint returning only component status.
- The dashboard container serves static files and proxies only `/api` and `/health`.

## 6. Dependency hygiene

- Dependencies are declared with ranges in `pyproject.toml` and installed from the CPU wheel index
  for PyTorch.
- CI installs from a clean environment on every push.
- `pip-audit`/`npm audit` are not wired into CI in v1.0.0; this is a known gap, recorded in
  `docs/IMPLEMENTATION_STATUS.md`.

## 7. Responsible disclosure

Report vulnerabilities privately to the repository owner rather than opening a public issue.
Include the affected endpoint, the reproduction steps and the observed impact.
