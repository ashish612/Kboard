# LangSmith Integration with LangGraph

This document provides an in-depth explanation of how LangSmith integrates with LangGraph projects, its architecture, and how to enable it in this project.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [How The Integration Works](#how-the-integration-works)
- [Key Concepts](#key-concepts)
- [Enabling LangSmith](#enabling-langsmith)
- [Advanced Features](#advanced-features)
- [Dashboard Features](#dashboard-features)

---

## Overview

LangSmith is LangChain's observability and evaluation platform. It provides **automatic tracing** for LangGraph applications with zero code changes — you only need to set environment variables.

### Key Benefits

- **Full Observability**: See every LLM call, chain execution, and state change
- **Debugging**: Trace errors back to specific nodes and inputs
- **Performance**: Track latency and token usage
- **Evaluation**: Create datasets and run automated evaluations
- **Feedback**: Collect human feedback on LLM outputs

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         YOUR APPLICATION                                 │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    LangGraph Workflow                             │   │
│  │   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐       │   │
│  │   │generate │───►│ review  │───►│  send   │───►│  webex  │       │   │
│  │   │ email   │    │  node   │    │  email  │    │  post   │       │   │
│  │   └────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘       │   │
│  │        │              │              │              │             │   │
│  │        ▼              ▼              ▼              ▼             │   │
│  │   ┌─────────────────────────────────────────────────────────────┐    │   │
│  │   │           LangChain Callbacks System                         │    │   │
│  │   │     (Automatically captures all operations)                  │    │   │
│  │   └───────────────────────────┬─────────────────────────────────┘    │   │
│  └───────────────────────────────┼──────────────────────────────────┘   │
│                                  │                                       │
│                                  ▼                                       │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │              LangSmith Tracer (langsmith.traceable)                │ │
│  │   - Run tree creation        - Token counting                      │ │
│  │   - Span management          - Latency tracking                    │ │
│  │   - Input/Output capture     - Error tracking                      │ │
│  └────────────────────────────────────┬───────────────────────────────┘ │
└───────────────────────────────────────┼─────────────────────────────────┘
                                        │ HTTPS
                                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       LANGSMITH CLOUD / SERVER                           │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐   │
│  │   Traces    │  │   Projects   │  │   Datasets   │  │ Evaluators  │   │
│  │   Storage   │  │   & Runs     │  │   & Testing  │  │ & Feedback  │   │
│  └─────────────┘  └──────────────┘  └──────────────┘  └─────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Component Breakdown

| Component | Role |
|-----------|------|
| **LangGraph Workflow** | Your application logic with nodes and edges |
| **Callbacks System** | LangChain's event system that fires on every operation |
| **LangSmith Tracer** | Captures callbacks and formats them for LangSmith |
| **LangSmith Cloud** | Stores, visualizes, and analyzes trace data |

---

## How The Integration Works

### 1. Environment Variable Auto-Configuration

LangSmith uses environment variables that LangChain/LangGraph automatically detect:

```bash
# Core configuration
LANGCHAIN_TRACING_V2=true          # Enable tracing
LANGCHAIN_API_KEY=your-api-key     # Your LangSmith API key
LANGCHAIN_PROJECT=bw-auto          # Project name in LangSmith
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com  # Default endpoint
```

When `LANGCHAIN_TRACING_V2=true`, LangChain's callback system automatically:

1. Creates a `LangChainTracer` instance
2. Registers it as a default callback
3. Sends trace data to LangSmith asynchronously

### 2. The Callbacks Architecture

The `CiscoBridgeChatModel` in this project extends `BaseChatModel`, which has built-in callback support:

```python
# From langchain_core/language_models/chat_models.py
class BaseChatModel:
    def invoke(self, input, config=None, **kwargs):
        # Callbacks automatically fire:
        # - on_llm_start: Before LLM call
        # - on_llm_end: After LLM call (with tokens, latency)
        # - on_llm_error: If error occurs
```

### 3. Callback Flow for This Workflow

```
graph.invoke(initial_state)
    │
    ├─► on_chain_start("CommunicationWorkflow")
    │
    ├─► Node: generate_email
    │       ├─► on_chain_start("ChatPromptTemplate")
    │       ├─► on_chain_end(formatted_prompt)
    │       │
    │       ├─► on_llm_start("CiscoBridgeChatModel")
    │       │       └─► Records: model, messages, temperature
    │       │
    │       └─► on_llm_end(response)
    │               └─► Records: output, tokens, latency
    │
    ├─► Node: email_review (interrupt)
    │       └─► on_chain_end(state with "awaiting_email_review")
    │
    ├─► [User resumes workflow]
    │
    ├─► Node: send_email
    │       └─► on_tool_start/end (if using LangChain tools)
    │
    └─► on_chain_end("CommunicationWorkflow", final_state)
```

### 4. LangGraph-Specific Tracing

LangGraph provides enhanced tracing that captures:

| Component | What's Traced |
|-----------|---------------|
| **Graph Execution** | Full run tree with parent-child relationships |
| **Nodes** | Each node as a child span with inputs/outputs |
| **Edges** | Conditional routing decisions |
| **Checkpoints** | State snapshots at interrupts |
| **State Changes** | Delta between node executions |

### 5. Example Trace Structure

For this project's workflow:

```
Run: CommunicationWorkflow
├── Node: generate_email
│   ├── Chain: ChatPromptTemplate
│   └── LLM: CiscoBridgeChatModel
│       ├── Input: {"message": "...", "sender_name": "..."}
│       ├── Output: "SUBJECT: ... ---\n..."
│       ├── Tokens: 150 in / 200 out
│       └── Latency: 1.2s
│
├── Node: email_review
│   └── State: awaiting_email_review (INTERRUPTED)
│
├── [Resume with email_approved=True]
│
├── Node: send_email
│   └── Custom span (if decorated)
│
├── Node: generate_webex_message
│   └── LLM: CiscoBridgeChatModel
│
└── Node: post_to_webex
    └── Final state: completed
```

---

## Key Concepts

### 1. Runs & Run Trees

Every execution creates a hierarchical "Run Tree":

```python
# Structure of a run
{
    "id": "uuid",
    "name": "CommunicationWorkflow",
    "run_type": "chain",  # or "llm", "tool", "retriever"
    "inputs": {...},
    "outputs": {...},
    "start_time": "...",
    "end_time": "...",
    "child_runs": [
        {"name": "generate_email", "run_type": "chain", ...},
        {"name": "CiscoBridgeChatModel", "run_type": "llm", ...}
    ]
}
```

### 2. Run Types

| Run Type | Description |
|----------|-------------|
| `chain` | Any chain or graph execution |
| `llm` | Language model calls |
| `tool` | Tool/function executions |
| `retriever` | Vector store retrievals |
| `embedding` | Embedding model calls |

### 3. Projects

Projects organize runs. Your `LANGCHAIN_PROJECT=bw-auto` groups all traces together for easy filtering and analysis.

### 4. Datasets & Evaluation

You can create test datasets and run evaluations:

```python
from langsmith import Client

client = Client()

# Create a dataset from production runs
dataset = client.create_dataset("email-generation-tests")

# Add examples
client.create_example(
    inputs={"message": "Hey team, servers are down"},
    outputs={"subject": "Urgent: Server Outage", "body": "..."},
    dataset_id=dataset.id
)
```

---

## Enabling LangSmith

### Step 1: Add Environment Variables

Add these to your `.env` file:

```bash
# LangSmith Configuration
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
LANGCHAIN_PROJECT=bw-auto

# Optional: Custom endpoint (for self-hosted)
# LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

### Step 2: Get Your API Key

1. Go to [smith.langchain.com](https://smith.langchain.com)
2. Sign up or log in
3. Go to Settings → API Keys
4. Create a new API key
5. Copy it to your `.env` file

### Step 3: (Optional) Update Config

Add LangSmith settings to `config.py`:

```python
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional

class Settings(BaseSettings):
    # ... existing fields ...
    
    # LangSmith Configuration
    langchain_tracing_v2: bool = Field(
        default=False, 
        description="Enable LangSmith tracing"
    )
    langchain_api_key: Optional[str] = Field(
        default=None, 
        description="LangSmith API key"
    )
    langchain_project: str = Field(
        default="bw-auto", 
        description="LangSmith project name"
    )
```

### Step 4: (Optional) Add Custom Tracing

For non-LangChain code (like email/webex services), use the `@traceable` decorator:

```python
from langsmith import traceable

class EmailService:
    @traceable(name="send_email", run_type="tool")
    def send_email(self, recipients, subject, body):
        # Your existing code - now traced!
        ...

class WebexService:
    @traceable(name="post_webex_message", run_type="tool")
    def post_message(self, room_id, text, markdown, mention_emails):
        # Your existing code - now traced!
        ...
```

### Step 5: Verify Integration

Run your workflow and check the LangSmith dashboard:

```bash
python main.py send "Test message for LangSmith" --no-review
```

Then visit [smith.langchain.com](https://smith.langchain.com) to see your traces.

---

## Advanced Features

### 1. Metadata & Tags

Add context to traces for filtering:

```python
# In workflow.py run method
config = {
    "configurable": {"thread_id": thread_id},
    "metadata": {
        "user_id": "user-123",
        "environment": "production",
        "version": "1.0.0"
    },
    "tags": ["email-workflow", "v1", "production"]
}

for event in self.graph.stream(initial_state, config, stream_mode="values"):
    result = event
```

### 2. Human Feedback

Collect feedback on LLM outputs:

```python
from langsmith import Client

client = Client()

# After user approves/rejects email
client.create_feedback(
    run_id="uuid-from-trace",
    key="user-approval",
    score=1.0 if approved else 0.0,
    comment=rejection_reason
)
```

### 3. Prompt Hub Integration

Version control your prompts:

```python
from langchain import hub

# Push a prompt
hub.push("bw-auto/email-generation", email_prompt)

# Pull a prompt (instead of hardcoding)
email_prompt = hub.pull("bw-auto/email-generation")
```

### 4. Custom Run Names

Make traces easier to identify:

```python
from langsmith import traceable

@traceable(name="generate_formal_email")
def generate_email(message: str, sender: str):
    ...
```

### 5. Async Tracing

LangSmith handles async operations:

```python
from langsmith import traceable

@traceable
async def async_operation():
    ...
```

### 6. Context Propagation

Traces automatically propagate through nested calls:

```python
@traceable(name="parent")
def parent():
    child()  # Automatically linked as child run

@traceable(name="child")
def child():
    ...
```

---

## Dashboard Features

Once enabled, your LangSmith dashboard will show:

### 1. Traces View

- Full execution tree for each workflow run
- Expandable nodes showing inputs/outputs
- Timing for each component
- Error highlighting

### 2. LLM Calls

- Token usage (input/output tokens)
- Latency metrics
- Full prompts and responses
- Model parameters

### 3. Errors

- Stack traces for failed runs
- Input state at failure point
- Error categorization

### 4. Interrupts

- Checkpoint states when human review is awaited
- Resume tracking

### 5. Analytics

- Token costs per run
- Latency percentiles
- Error rates
- Usage trends

### 6. Filtering & Search

- Filter by project, tags, metadata
- Search by input/output content
- Time range selection

---

## Comparison: With vs Without LangSmith

| Aspect | Without LangSmith | With LangSmith |
|--------|-------------------|----------------|
| **Debugging** | Print statements, logs | Full trace visualization |
| **Token Tracking** | Manual counting | Automatic |
| **Latency** | Manual timing | Per-component breakdown |
| **Errors** | Log parsing | Visual error traces |
| **Evaluation** | Manual testing | Automated evaluation |
| **Feedback** | Not tracked | Systematically collected |

---

## Security Considerations

### Data Sent to LangSmith

- LLM inputs and outputs
- Token counts
- Latency data
- Error messages
- Custom metadata

### Sensitive Data

⚠️ **Warning**: All LLM inputs/outputs are sent to LangSmith. Consider:

1. **Masking**: Use LangSmith's data masking features
2. **Self-Hosting**: Deploy LangSmith on your infrastructure
3. **Filtering**: Don't trace sensitive operations

```python
# Disable tracing for sensitive operations
from langchain_core.tracers.context import tracing_v2_enabled

with tracing_v2_enabled(enabled=False):
    # This won't be traced
    sensitive_operation()
```

---

## Troubleshooting

### Traces Not Appearing

1. Verify `LANGCHAIN_TRACING_V2=true`
2. Check API key is valid
3. Ensure network access to api.smith.langchain.com
4. Check for rate limiting

### Missing Child Runs

1. Ensure callbacks are not disabled
2. Check that operations use LangChain components
3. Add `@traceable` to custom functions

### High Latency

1. Tracing is async by default
2. Check network latency to LangSmith
3. Consider batching for high-volume scenarios

---

## Summary

| Aspect | Detail |
|--------|--------|
| **Integration Type** | Zero-code (env vars) + optional decorators |
| **What's Traced** | LLM calls, chain runs, graph nodes, state changes |
| **Data Captured** | Inputs, outputs, tokens, latency, errors |
| **Custom LLM Compatibility** | Works via `BaseChatModel` inheritance |
| **Custom Services** | Use `@traceable` decorator |
| **Dashboard** | smith.langchain.com |

---

## Infrastructure Requirements

LangSmith offers three deployment models to accommodate different infrastructure needs.

### Deployment Options

#### 1. Cloud (SaaS) - Zero Infrastructure

| Aspect | Details |
|--------|---------|
| **Managed By** | LangChain (fully hosted) |
| **Infrastructure Required** | None |
| **URL** | https://smith.langchain.com |
| **Setup** | Just set environment variables |
| **Best For** | Quick start, small-medium teams, development |

```bash
# Only these env vars needed
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_xxxxx
LANGCHAIN_PROJECT=bw-auto
```

#### 2. Hybrid Deployment

| Aspect | Details |
|--------|---------|
| **Control Plane** | Managed by LangChain |
| **Data Plane** | Hosted in your infrastructure |
| **Data Residency** | Your data stays in your environment |
| **Best For** | Enterprise with data sovereignty requirements |

#### 3. Self-Hosted - Full Infrastructure

For complete control, you run the entire LangSmith stack.

##### Compute Resources

| Component | CPU | RAM | Purpose |
|-----------|-----|-----|---------|
| **Frontend** | 1 | 2 GB | Web UI |
| **Backend** | 2 | 4 GB | Core API services |
| **Platform Backend** | 1 | 2 GB | Platform APIs |
| **Queue** | 1 | 2 GB | Async job processing |
| **Playground** | 1 | 2 GB | LLM testing interface |
| **ACE Backend** | 1 | 2 GB | Analytics & evaluation |
| **PostgreSQL** | 2 | 8 GB | Operational data |
| **Redis** | 2 | 4 GB | Caching & queuing |
| **ClickHouse** | 8 | 32 GB | Traces & analytics |
| **Total Minimum** | **19** | **58 GB** | |

##### Storage Services

```
┌─────────────────────────────────────────────────────────────────┐
│                    LangSmith Self-Hosted                         │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │  PostgreSQL  │  │    Redis     │  │     ClickHouse       │   │
│  │              │  │              │  │                      │   │
│  │ - Users      │  │ - Sessions   │  │ - Traces (billions)  │   │
│  │ - Projects   │  │ - Queues     │  │ - Feedback           │   │
│  │ - API Keys   │  │ - Cache      │  │ - Analytics          │   │
│  │ - Datasets   │  │              │  │ - Time-series data   │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              Blob Storage (Optional)                        │ │
│  │   - S3 / GCS / Azure Blob / MinIO                          │ │
│  │   - Large trace payloads                                    │ │
│  │   - Attachments                                             │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

##### Networking Requirements

| Direction | Endpoint | Purpose |
|-----------|----------|---------|
| **Egress** | `https://beacon.langchain.com` | License verification |
| **Ingress** | Your domain | User access to UI |
| **Internal** | Between services | Service communication |

##### Deployment Methods

**Docker Compose** (Development/Testing):
```bash
git clone https://github.com/langchain-ai/langsmith-sdk
cd langsmith-sdk/langsmith
docker-compose up -d
```

**Kubernetes/Helm** (Production):
```bash
helm repo add langchain https://langchain-ai.github.io/helm/
helm install langsmith langchain/langsmith
```

### Deployment Comparison

| Feature | Cloud | Hybrid | Self-Hosted |
|---------|-------|--------|-------------|
| **Infrastructure Cost** | $0 | Medium | High |
| **Setup Complexity** | None | Medium | High |
| **Data Location** | LangChain Cloud | Your infra | Your infra |
| **Maintenance** | None | Partial | Full |
| **Customization** | Limited | Medium | Full |
| **Pricing** | Usage-based | Enterprise | License |
| **Min Resources** | - | ~30 GB RAM | ~60 GB RAM |

### Recommendation for This Project

For the **bw-auto** project, we recommend starting with the **Cloud (SaaS)** option:

1. **Zero infrastructure overhead** - No servers to manage
2. **Instant setup** - Just add 3 environment variables
3. **Free tier available** - Generous limits for development
4. **Easy migration** - Can move to self-hosted later if needed

If data sovereignty or compliance requirements dictate, consider the **Hybrid** or **Self-Hosted** options.

---

## References

- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [LangSmith Self-Hosting Guide](https://docs.smith.langchain.com/self_hosting/)
- [LangSmith Docker Deployment](https://docs.langchain.com/langsmith/docker)
- [LangGraph Tracing](https://langchain-ai.github.io/langgraph/concepts/low_level/#tracing)
- [LangChain Callbacks](https://python.langchain.com/docs/concepts/callbacks/)
- [LangSmith Python SDK](https://github.com/langchain-ai/langsmith-sdk)

