"""A realistic prior store, injected into every system before the transcript.

Why this exists: with only the transcript's own facts a store holds two to four rows,
and "the search returned every row" is true but uninteresting when the default result
cap is 20. Forty prior facts make returning all of it a choice rather than an artefact
of its size.

They are deliberately the same user's real work. That is the hard case: on an ordinary
turn the store is full of things that look related to whatever was just said, so scores
stay high enough to look plausible. A distractor set about unrelated topics would
flatter every system.

Each fact carries a subject and an attribute as well as its text. Most systems ignore
them and store the string alone; Graphiti and AgentCore re-extract into their own
shapes. They are carried here so that a system keyed on entity plus attribute can be
given a well-formed write without changing the text any other system sees.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Distractor:
    """One prior fact: what it is about, which property of it, and the text itself."""

    subject: str
    kind: str
    attribute: str
    text: str


DISTRACTORS: list[Distractor] = [
    Distractor("checkout service", "product", "runtime", "The checkout service runs on ECS Fargate in ap-south-1."),
    Distractor("payments database", "product", "engine", "The payments database is Postgres 16 on RDS with a read replica."),
    Distractor("Redis", "product", "use", "Redis is used for session storage and rate limiting."),
    Distractor("outbound webhooks", "product", "retry_policy", "The retry policy for outbound webhooks uses exponential backoff."),
    Distractor("feature flags", "product", "provider", "Feature flags are managed in LaunchDarkly."),
    Distractor("staging environment", "project", "capacity", "The staging environment mirrors production at a quarter of the capacity."),
    Distractor("deploy pipeline", "project", "trigger", "Deploys go out through GitHub Actions on merge to main."),
    Distractor("API gateway", "product", "timeout", "The API gateway enforces a 30 second request timeout."),
    Distractor("background jobs", "product", "runner", "Background jobs run on Celery with RabbitMQ as the broker."),
    Distractor("observability stack", "product", "components", "Observability is Grafana dashboards on top of Prometheus."),
    Distractor("search index", "product", "engine", "The search index is OpenSearch, reindexed nightly."),
    Distractor("object storage", "product", "lifecycle", "Object storage is S3 with lifecycle rules moving old objects to Glacier."),
    Distractor("February outage", "project", "cause", "A checkout outage in February was caused by connection pool exhaustion."),
    Distractor("March incident review", "project", "recommendation", "The March incident review recommended adding circuit breakers."),
    Distractor("retry storm", "project", "impact", "An earlier retry storm doubled load on the payments service."),
    Distractor("Kafka migration", "project", "status", "The team postponed the Kafka migration to next quarter."),
    Distractor("notification worker", "product", "leak_fix", "A memory leak in the notification worker was fixed by upgrading the SDK."),
    Distractor("July load test", "project", "peak", "The load test in July peaked at 4K requests per second."),
    Distractor("Priya", "person", "role", "Priya leads the platform team."),
    Distractor("Karthik", "person", "role", "Karthik reviews all database migrations."),
    Distractor("Sneha", "person", "role", "Sneha manages the on-call rotation."),
    Distractor("Vikram", "person", "role", "Vikram works on the mobile client."),
    Distractor("Ananya", "person", "role", "The design system is maintained by Ananya."),
    Distractor("Meera", "person", "role", "Meera is the product manager for checkout."),
    Distractor("code review", "project", "approval_policy", "Code review requires one approval before merge."),
    Distractor("database migrations", "project", "compatibility_policy", "Migrations must be backwards compatible for one release."),
    Distractor("alerts", "project", "runbook_policy", "The team writes runbooks for every alert."),
    Distractor("documentation", "repo", "location", "Documentation lives in the docs directory of each repository."),
    Distractor("secrets", "project", "storage_policy", "Secrets are stored in AWS Secrets Manager, never in environment files."),
    Distractor("Python code", "repo", "type_hint_policy", "Type hints are required on all new Python code."),
    Distractor("Python code", "repo", "line_length", "Line length is capped at 100 characters."),
    Distractor("dependencies", "repo", "update_cadence", "Dependency updates are batched weekly."),
    Distractor("local development", "repo", "tooling", "Local development uses Docker Compose."),
    Distractor("dependencies", "repo", "manager", "Dependencies are managed with uv."),
    Distractor("Python code", "repo", "linter", "Linting is ruff, formatting is ruff format."),
    Distractor("Python code", "repo", "type_checker", "Type checking is mypy in strict mode."),
    Distractor("monorepo", "repo", "layout", "The monorepo is split into services and packages directories."),
    Distractor("cloud infrastructure", "project", "tooling", "Terraform manages all cloud infrastructure."),
    Distractor("alerts", "project", "routing", "Alerting routes through PagerDuty."),
    Distractor("changelog", "repo", "source", "The changelog is generated from conventional commit messages."),
]

TEXTS: list[str] = [d.text for d in DISTRACTORS]

assert len(DISTRACTORS) == 40, len(DISTRACTORS)
assert len({(d.subject, d.attribute) for d in DISTRACTORS}) == 40, "subject+attribute must be unique"
