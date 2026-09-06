# AgentCore setup

What is needed before `runners/run_agentcore.py` can run. Everything here is done once,
in a personal AWS account.

## Cost

Under one dollar for the whole experiment, and there is no free tier.

| What | Rate | This experiment |
| --- | --- | --- |
| Short-term memory events | $0.25 per 1,000 | ~130 events across 3 runs, about $0.03 |
| Long-term records, built-in strategies | $0.75 per 1,000 per month | ~150 records, about $0.11 per month |
| Retrievals | $0.50 per 1,000 | ~15 retrievals, under $0.01 |

The runner deletes its memory resource at the end of each run, so the monthly storage
line stops accruing as soon as the runs finish. If a run crashes before cleanup, delete
leftovers with the command at the bottom of this file.

## 1. Pick a region

AgentCore Memory is in **us-east-1** and **us-west-2** with full feature support, plus a
handful of Memory-only regions (Paris, Seoul, Stockholm, London, Canada Central, São
Paulo). Use `us-east-1` unless there is a reason not to.

## 2. Create an IAM user with just these permissions

No execution role is needed. `memoryExecutionRoleArn` is optional on `CreateMemory`, and
the built-in extraction strategies are service-managed, so they do not invoke a model in
your account and you do not need to request Bedrock model access.

In the AWS console: **IAM → Users → Create user**, name it `agent-memory-experiments`,
skip console access, and attach an inline policy with this JSON:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AgentCoreMemoryControlPlane",
      "Effect": "Allow",
      "Action": [
        "bedrock-agentcore:CreateMemory",
        "bedrock-agentcore:GetMemory",
        "bedrock-agentcore:UpdateMemory",
        "bedrock-agentcore:DeleteMemory",
        "bedrock-agentcore:ListMemories"
      ],
      "Resource": "*"
    },
    {
      "Sid": "AgentCoreMemoryDataPlane",
      "Effect": "Allow",
      "Action": [
        "bedrock-agentcore:CreateEvent",
        "bedrock-agentcore:GetEvent",
        "bedrock-agentcore:ListEvents",
        "bedrock-agentcore:DeleteEvent",
        "bedrock-agentcore:ListSessions",
        "bedrock-agentcore:RetrieveMemoryRecords",
        "bedrock-agentcore:ListMemoryRecords",
        "bedrock-agentcore:GetMemoryRecord",
        "bedrock-agentcore:DeleteMemoryRecord",
        "bedrock-agentcore:BatchCreateMemoryRecords",
        "bedrock-agentcore:StartMemoryExtractionJob",
        "bedrock-agentcore:ListMemoryExtractionJobs"
      ],
      "Resource": "*"
    }
  ]
}
```

`Resource: "*"` is fine here because the account exists for this experiment. Narrow it to
the memory ARN if you reuse the account for anything else.

## 3. Create an access key

**IAM → Users → agent-memory-experiments → Security credentials → Create access key**,
choose "Application running outside AWS". Copy both values; the secret is shown once.

## 4. Add a named profile

Append to `~/.aws/credentials`:

```ini
[personal-agentcore]
aws_access_key_id = AKIA...
aws_secret_access_key = ...
```

And to `~/.aws/config`:

```ini
[profile personal-agentcore]
region = us-east-1
output = json
```

The profile name matters: the runner reads `AMX_AWS_PROFILE`, defaulting to
`personal-agentcore`, so nothing here can accidentally reach a work account.

## 5. Verify

```bash
aws sts get-caller-identity --profile personal-agentcore
aws bedrock-agentcore-control list-memories --profile personal-agentcore --region us-east-1
```

The first prints the account id. The second should return an empty list rather than an
`AccessDenied`.

## Cleaning up leftovers

```bash
aws bedrock-agentcore-control list-memories --profile personal-agentcore --region us-east-1
aws bedrock-agentcore-control delete-memory --memory-id <id> --profile personal-agentcore --region us-east-1
```
