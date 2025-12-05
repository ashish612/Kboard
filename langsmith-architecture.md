# LangSmith-Type System Architecture

LangSmith is an LLM application observability, debugging, testing, and evaluation platform. This document provides a comprehensive architecture for implementing a similar system.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  SDK/Instrumentation  │  Web Dashboard  │  CLI Tools  │  API Clients        │
│  (Python, JS, etc.)   │  (React/Vue)    │             │                     │
└─────────────┬─────────────────┬─────────────────┬─────────────────┬─────────┘
              │                 │                 │                 │
              ▼                 ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            API GATEWAY                                       │
│  - Authentication/Authorization (OAuth2, API Keys)                          │
│  - Rate Limiting                                                            │
│  - Request Routing                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          INGESTION LAYER                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │ Trace Collector │  │ Event Processor │  │ Batch Processor │              │
│  │                 │  │                 │  │                 │              │
│  │ - Async ingestion│  │ - Validation   │  │ - Bulk writes   │              │
│  │ - Buffering     │  │ - Enrichment   │  │ - Compression   │              │
│  │ - Sampling      │  │ - Normalization│  │                 │              │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘              │
└───────────┼────────────────────┼────────────────────┼────────────────────────┘
            │                    │                    │
            ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MESSAGE QUEUE (Kafka/RabbitMQ)                        │
│  - traces-topic  │  - evaluations-topic  │  - feedback-topic                │
└─────────────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          CORE SERVICES                                       │
├──────────────────┬──────────────────┬──────────────────┬────────────────────┤
│  Tracing Service │ Evaluation Svc   │ Dataset Service  │ Feedback Service   │
│                  │                  │                  │                    │
│  - Run tracking  │ - Auto-eval      │ - CRUD datasets  │ - Human feedback   │
│  - Span mgmt     │ - Custom evals   │ - Versioning     │ - Annotations      │
│  - Tree assembly │ - LLM-as-judge   │ - Splits (train/ │ - Corrections      │
│  - Latency calc  │ - Scoring        │   test/val)      │ - Ratings          │
└──────────────────┴──────────────────┴──────────────────┴────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         STORAGE LAYER                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │ PostgreSQL      │  │ ClickHouse/     │  │ Object Storage  │              │
│  │                 │  │ TimescaleDB     │  │ (S3/MinIO)      │              │
│  │ - Projects      │  │                 │  │                 │              │
│  │ - Datasets      │  │ - Traces        │  │ - Large payloads│              │
│  │ - Users/Orgs    │  │ - Metrics       │  │ - Artifacts     │              │
│  │ - Evaluators    │  │ - Aggregations  │  │ - Exports       │              │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘              │
│                                                                              │
│  ┌─────────────────┐  ┌─────────────────┐                                   │
│  │ Redis           │  │ Elasticsearch   │                                   │
│  │                 │  │                 │                                   │
│  │ - Caching       │  │ - Full-text     │                                   │
│  │ - Sessions      │  │   search        │                                   │
│  │ - Rate limits   │  │ - Log search    │                                   │
│  └─────────────────┘  └─────────────────┘                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Data Models

### 1. Trace/Run Model (The Heart of Observability)

```python
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from uuid import UUID

class RunType(Enum):
    CHAIN = "chain"
    LLM = "llm"
    TOOL = "tool"
    RETRIEVER = "retriever"
    EMBEDDING = "embedding"
    PARSER = "parser"
    PROMPT = "prompt"

class RunStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"

@dataclass
class Run:
    """A single execution unit (span) in a trace"""
    id: UUID
    trace_id: UUID                    # Groups all runs in a single request
    parent_run_id: Optional[UUID]     # For nested runs (tree structure)
    
    # Identification
    name: str                         # e.g., "ChatOpenAI", "RetrievalQA"
    run_type: RunType
    project_id: UUID
    
    # Timing
    start_time: datetime
    end_time: Optional[datetime]
    
    # Status
    status: RunStatus
    error: Optional[str]
    
    # I/O Capture
    inputs: Dict[str, Any]            # Serialized inputs
    outputs: Optional[Dict[str, Any]] # Serialized outputs
    
    # LLM-specific
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    model_name: Optional[str]
    
    # Metadata
    extra: Dict[str, Any]             # Tags, metadata, etc.
    session_id: Optional[str]         # For conversation grouping
    
    # Computed
    latency_ms: Optional[int]
    
    # Feedback/Eval references
    feedback_ids: List[UUID] = None
```

### 2. Dataset & Example Models

```python
@dataclass
class Dataset:
    """Collection of examples for testing/evaluation"""
    id: UUID
    name: str
    description: Optional[str]
    project_id: UUID
    
    # Versioning
    version: int
    created_at: datetime
    updated_at: datetime
    
    # Schema definition (optional)
    input_schema: Optional[Dict[str, Any]]
    output_schema: Optional[Dict[str, Any]]

@dataclass
class Example:
    """Single test case in a dataset"""
    id: UUID
    dataset_id: UUID
    
    inputs: Dict[str, Any]
    outputs: Optional[Dict[str, Any]]  # Expected/reference outputs
    
    # Metadata
    metadata: Dict[str, Any]
    split: str = "base"  # train, test, validation, base
    
    # Source tracking
    source_run_id: Optional[UUID]  # If created from a production trace
    created_at: datetime
```

### 3. Evaluation Models

```python
@dataclass
class Evaluator:
    """Definition of an evaluation function"""
    id: UUID
    name: str
    description: str
    project_id: UUID
    
    # Type
    evaluator_type: str  # "llm", "code", "human", "heuristic"
    
    # Configuration
    config: Dict[str, Any]  # Model, prompt template, code, etc.
    
    # Output schema
    score_type: str  # "binary", "continuous", "categorical"
    score_config: Dict[str, Any]  # min, max, categories, etc.

@dataclass
class EvaluationResult:
    """Result of running an evaluator on a run"""
    id: UUID
    run_id: UUID
    evaluator_id: UUID
    
    score: float
    value: Optional[str]  # For categorical
    comment: Optional[str]
    
    # LLM evaluator reasoning
    reasoning: Optional[str]
    
    created_at: datetime
    
@dataclass
class Feedback:
    """Human feedback on a run"""
    id: UUID
    run_id: UUID
    
    # Feedback type
    key: str  # e.g., "correctness", "helpfulness", "user_rating"
    score: Optional[float]
    value: Optional[str]
    comment: Optional[str]
    
    # Source
    source_type: str  # "human", "model", "api"
    source_user_id: Optional[UUID]
    
    created_at: datetime
```

---

## Component Deep Dives

### 1. SDK/Instrumentation Layer

```python
# Example SDK interface (Python)
import asyncio
from contextlib import contextmanager
from functools import wraps
import time
from typing import Callable, Any
import httpx
from uuid import uuid4

class TracingClient:
    def __init__(self, api_key: str, project: str, endpoint: str = "https://api.yourplatform.com"):
        self.api_key = api_key
        self.project = project
        self.endpoint = endpoint
        self._buffer: list = []
        self._flush_interval = 1.0  # seconds
        self._batch_size = 100
        
    @contextmanager
    def trace(self, name: str, run_type: str = "chain", **kwargs):
        """Context manager for tracing a block of code"""
        run_id = uuid4()
        trace_id = kwargs.get("trace_id") or uuid4()
        parent_id = kwargs.get("parent_run_id")
        
        run = {
            "id": str(run_id),
            "trace_id": str(trace_id),
            "parent_run_id": str(parent_id) if parent_id else None,
            "name": name,
            "run_type": run_type,
            "start_time": time.time(),
            "inputs": kwargs.get("inputs", {}),
            "status": "running",
        }
        
        try:
            yield run
            run["status"] = "success"
        except Exception as e:
            run["status"] = "error"
            run["error"] = str(e)
            raise
        finally:
            run["end_time"] = time.time()
            run["latency_ms"] = int((run["end_time"] - run["start_time"]) * 1000)
            self._buffer.append(run)
            self._maybe_flush()
    
    def traceable(self, name: str = None, run_type: str = "chain"):
        """Decorator for automatic tracing"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                trace_name = name or func.__name__
                with self.trace(trace_name, run_type, inputs={"args": args, "kwargs": kwargs}) as run:
                    result = func(*args, **kwargs)
                    run["outputs"] = {"result": result}
                    return result
            return wrapper
        return decorator
    
    def _maybe_flush(self):
        if len(self._buffer) >= self._batch_size:
            self._flush()
    
    def _flush(self):
        if not self._buffer:
            return
        # Async batch send to API
        runs = self._buffer.copy()
        self._buffer.clear()
        # Send to ingestion endpoint (async in production)
        httpx.post(
            f"{self.endpoint}/api/v1/runs/batch",
            json={"runs": runs},
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
```

### 2. Ingestion Service Architecture

```python
# FastAPI-based ingestion service
from fastapi import FastAPI, BackgroundTasks, Depends
from pydantic import BaseModel
from typing import List
import aiokafka

app = FastAPI()

class RunCreate(BaseModel):
    id: str
    trace_id: str
    parent_run_id: str | None
    name: str
    run_type: str
    start_time: float
    end_time: float | None
    inputs: dict
    outputs: dict | None
    status: str
    error: str | None

class BatchRunRequest(BaseModel):
    runs: List[RunCreate]

# Kafka producer for async processing
producer: aiokafka.AIOKafkaProducer = None

@app.on_event("startup")
async def startup():
    global producer
    producer = aiokafka.AIOKafkaProducer(
        bootstrap_servers='kafka:9092',
        value_serializer=lambda v: json.dumps(v).encode()
    )
    await producer.start()

@app.post("/api/v1/runs/batch")
async def ingest_runs(request: BatchRunRequest, background_tasks: BackgroundTasks):
    """High-throughput batch ingestion endpoint"""
    
    # Validate and enrich
    enriched_runs = []
    for run in request.runs:
        enriched = run.dict()
        enriched["ingested_at"] = time.time()
        enriched["project_id"] = get_project_from_context()
        enriched_runs.append(enriched)
    
    # Send to Kafka for async processing
    for run in enriched_runs:
        await producer.send("traces-topic", run)
    
    return {"status": "accepted", "count": len(enriched_runs)}
```

### 3. Trace Processing Worker

```python
# Worker that processes traces from Kafka
import asyncio
from aiokafka import AIOKafkaConsumer
import clickhouse_connect

class TraceProcessor:
    def __init__(self):
        self.consumer = None
        self.ch_client = clickhouse_connect.get_client(host='clickhouse')
        
    async def start(self):
        self.consumer = AIOKafkaConsumer(
            'traces-topic',
            bootstrap_servers='kafka:9092',
            group_id='trace-processor',
            value_deserializer=lambda m: json.loads(m.decode())
        )
        await self.consumer.start()
        
        async for msg in self.consumer:
            await self.process_run(msg.value)
    
    async def process_run(self, run: dict):
        # 1. Store in ClickHouse for analytics
        self.ch_client.insert(
            'runs',
            [run],
            column_names=list(run.keys())
        )
        
        # 2. Update trace tree in Redis for real-time UI
        await self.update_trace_tree(run)
        
        # 3. Trigger auto-evaluations if configured
        await self.trigger_evaluations(run)
        
        # 4. Check for alerts/anomalies
        await self.check_alerts(run)
    
    async def update_trace_tree(self, run: dict):
        """Build hierarchical trace structure in Redis"""
        trace_key = f"trace:{run['trace_id']}"
        # Store run and update parent-child relationships
        await redis.hset(trace_key, run['id'], json.dumps(run))
        if run.get('parent_run_id'):
            await redis.sadd(f"children:{run['parent_run_id']}", run['id'])
```

### 4. Evaluation Engine

```python
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseEvaluator(ABC):
    """Base class for all evaluators"""
    
    @abstractmethod
    async def evaluate(self, run: Run) -> EvaluationResult:
        pass

class LLMJudgeEvaluator(BaseEvaluator):
    """Uses an LLM to evaluate outputs"""
    
    def __init__(self, model: str, prompt_template: str, criteria: List[str]):
        self.model = model
        self.prompt_template = prompt_template
        self.criteria = criteria
    
    async def evaluate(self, run: Run) -> EvaluationResult:
        prompt = self.prompt_template.format(
            input=run.inputs,
            output=run.outputs,
            criteria=self.criteria
        )
        
        # Call LLM for evaluation
        response = await llm_client.generate(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.content)
        return EvaluationResult(
            run_id=run.id,
            score=result["score"],
            reasoning=result["reasoning"]
        )

class HeuristicEvaluator(BaseEvaluator):
    """Code-based evaluators"""
    
    def __init__(self, eval_fn: Callable):
        self.eval_fn = eval_fn
    
    async def evaluate(self, run: Run) -> EvaluationResult:
        score = self.eval_fn(run.inputs, run.outputs)
        return EvaluationResult(run_id=run.id, score=score)

# Built-in evaluators
def string_match_evaluator(inputs: dict, outputs: dict) -> float:
    """Exact string match"""
    expected = inputs.get("expected_output", "")
    actual = outputs.get("output", "")
    return 1.0 if expected == actual else 0.0

def contains_evaluator(inputs: dict, outputs: dict) -> float:
    """Check if output contains expected substring"""
    expected = inputs.get("must_contain", "")
    actual = outputs.get("output", "")
    return 1.0 if expected in actual else 0.0

def json_validity_evaluator(inputs: dict, outputs: dict) -> float:
    """Check if output is valid JSON"""
    try:
        json.loads(outputs.get("output", ""))
        return 1.0
    except:
        return 0.0
```

---

## Database Schema (ClickHouse for Analytics)

```sql
-- Main runs table (high-volume, time-series optimized)
CREATE TABLE runs (
    id UUID,
    trace_id UUID,
    parent_run_id Nullable(UUID),
    project_id UUID,
    
    name String,
    run_type LowCardinality(String),
    status LowCardinality(String),
    
    start_time DateTime64(3),
    end_time Nullable(DateTime64(3)),
    latency_ms Nullable(UInt32),
    
    -- Tokenize as JSON for flexible querying
    inputs String,  -- JSON
    outputs String, -- JSON
    error Nullable(String),
    
    -- LLM specifics
    model_name Nullable(String),
    prompt_tokens Nullable(UInt32),
    completion_tokens Nullable(UInt32),
    total_tokens Nullable(UInt32),
    
    -- Metadata
    session_id Nullable(String),
    extra String, -- JSON
    
    -- Ingestion metadata
    ingested_at DateTime64(3) DEFAULT now64(3)
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(start_time)
ORDER BY (project_id, trace_id, start_time, id)
TTL start_time + INTERVAL 90 DAY;

-- Materialized view for aggregations
CREATE MATERIALIZED VIEW run_stats_mv
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMMDD(start_time)
ORDER BY (project_id, run_type, model_name, toStartOfHour(start_time))
AS SELECT
    project_id,
    run_type,
    model_name,
    toStartOfHour(start_time) as hour,
    count() as run_count,
    sum(latency_ms) as total_latency_ms,
    sum(total_tokens) as total_tokens,
    countIf(status = 'error') as error_count
FROM runs
GROUP BY project_id, run_type, model_name, hour;

-- Feedback table
CREATE TABLE feedback (
    id UUID,
    run_id UUID,
    project_id UUID,
    
    key LowCardinality(String),
    score Nullable(Float64),
    value Nullable(String),
    comment Nullable(String),
    
    source_type LowCardinality(String),
    source_user_id Nullable(UUID),
    
    created_at DateTime64(3) DEFAULT now64(3)
)
ENGINE = MergeTree()
ORDER BY (project_id, run_id, created_at);
```

---

## API Design

```yaml
# OpenAPI spec highlights
paths:
  # Runs/Traces
  /api/v1/runs:
    post:
      summary: Create a single run
    get:
      summary: List runs with filtering
      parameters:
        - name: project_id
        - name: trace_id
        - name: run_type
        - name: status
        - name: start_time_gte
        - name: start_time_lte
        - name: session_id
        
  /api/v1/runs/batch:
    post:
      summary: Batch create runs (high throughput)
      
  /api/v1/runs/{run_id}:
    get:
      summary: Get run details
    patch:
      summary: Update run (add outputs, end time)
      
  /api/v1/traces/{trace_id}:
    get:
      summary: Get full trace tree
      
  # Datasets
  /api/v1/datasets:
    get:
      summary: List datasets
    post:
      summary: Create dataset
      
  /api/v1/datasets/{dataset_id}/examples:
    get:
      summary: List examples
    post:
      summary: Add examples (single or batch)
      
  # Evaluations
  /api/v1/evaluations:
    post:
      summary: Run evaluation on dataset
      requestBody:
        content:
          application/json:
            schema:
              properties:
                dataset_id: string
                evaluator_ids: array
                target_function: string  # or endpoint
                
  /api/v1/feedback:
    post:
      summary: Submit feedback for a run
      
  # Analytics
  /api/v1/analytics/runs:
    get:
      summary: Get run statistics
      parameters:
        - name: group_by  # hour, day, run_type, model
        - name: metrics   # count, latency_p50, latency_p99, tokens
```

---

## Recommended Tech Stack

| Component | Recommended Tech | Alternatives |
|-----------|-----------------|--------------|
| **API Gateway** | Kong / AWS API Gateway | Traefik, Envoy |
| **Backend Services** | FastAPI (Python) | Go, Node.js |
| **Message Queue** | Apache Kafka | RabbitMQ, AWS SQS |
| **Time-series DB** | ClickHouse | TimescaleDB, Apache Druid |
| **Relational DB** | PostgreSQL | CockroachDB |
| **Cache** | Redis | Memcached |
| **Search** | Elasticsearch | OpenSearch, Meilisearch |
| **Object Storage** | MinIO / S3 | GCS, Azure Blob |
| **Frontend** | React + TanStack Query | Vue, Svelte |
| **Visualization** | Tremor, Recharts | D3.js, Plotly |

---

## Key Features to Implement

### Phase 1: Core Observability
- [ ] Run/trace ingestion and storage
- [ ] Hierarchical trace visualization
- [ ] Basic filtering and search
- [ ] Token/latency metrics

### Phase 2: Datasets & Testing
- [ ] Dataset CRUD
- [ ] Example management
- [ ] Test run execution
- [ ] Comparison views

### Phase 3: Evaluation
- [ ] Built-in evaluators
- [ ] LLM-as-judge
- [ ] Custom evaluator definitions
- [ ] Evaluation experiments

### Phase 4: Advanced
- [ ] Prompt versioning & management
- [ ] A/B testing framework
- [ ] Alerting & anomaly detection
- [ ] Playground/debugging tools

---

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Kubernetes Cluster                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Ingress     │  │ API Pods    │  │ Worker Pods │             │
│  │ Controller  │  │ (HPA)       │  │ (HPA)       │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Kafka       │  │ ClickHouse  │  │ PostgreSQL  │             │
│  │ (StatefulSet)│ │ (Operator)  │  │ (Operator)  │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐                              │
│  │ Redis       │  │ MinIO       │                              │
│  │ (Cluster)   │  │ (S3-compat) │                              │
│  └─────────────┘  └─────────────┘                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Security Considerations

1. **API Authentication**: Use API keys with scoped permissions per project
2. **Data Encryption**: Encrypt sensitive data at rest and in transit
3. **PII Handling**: Implement automatic PII detection and masking in traces
4. **Access Control**: RBAC for projects, datasets, and evaluations
5. **Audit Logging**: Track all data access and modifications

