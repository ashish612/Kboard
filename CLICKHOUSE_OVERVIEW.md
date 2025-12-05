# ClickHouse Overview

**ClickHouse** is an open-source, high-performance **column-oriented database management system (DBMS)** designed specifically for **Online Analytical Processing (OLAP)**. It was originally developed by Yandex in 2009 to power Yandex.Metrica (one of the world's largest web analytics platforms handling billions of events per day) and was open-sourced under the Apache 2.0 license in 2016.

---

## Core Architecture & How It Works

### 1. Columnar Storage

Unlike row-based databases, ClickHouse stores data by columns rather than rows:

```
Row-based:        Column-based (ClickHouse):
┌─────────────┐   ┌────────────┐ ┌────────────┐ ┌────────────┐
│ id,ts,value │   │ id column  │ │ ts column  │ │val column  │
│ id,ts,value │   │ 1,2,3,4... │ │ t1,t2,t3...│ │ v1,v2,v3...│
│ id,ts,value │   └────────────┘ └────────────┘ └────────────┘
└─────────────┘
```

**Benefits:**
- Queries only read columns they need (not entire rows)
- Excellent compression ratios (similar values compress well)
- Vectorized query execution (SIMD operations on column batches)

### 2. MergeTree Engine Family

The primary table engine for most use cases:

```sql
CREATE TABLE events (
    timestamp DateTime,
    event_type String,
    user_id UInt64,
    duration_ms UInt32
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (event_type, timestamp)
```

- **Partitioning**: Data is divided into partitions (e.g., by month)
- **Primary Index**: Sparse index based on `ORDER BY` for fast range lookups
- **Background Merges**: Small data parts are merged asynchronously

---

## ClickHouse for Time Series Data

ClickHouse excels at time series workloads due to:

### 1. Efficient Time-Based Partitioning

```sql
-- Partition by day for time series
PARTITION BY toYYYYMMDD(timestamp)
```

### 2. Built-in Time Series Functions

```sql
-- Window functions for time series
SELECT 
    timestamp,
    value,
    avg(value) OVER (ORDER BY timestamp ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) as moving_avg
FROM metrics;

-- Time bucket aggregations
SELECT 
    toStartOfHour(timestamp) as hour,
    avg(cpu_usage),
    max(memory_usage)
FROM system_metrics
GROUP BY hour
ORDER BY hour;
```

### 3. Specialized Data Types

- `DateTime64` with sub-second precision
- `Date32` for extended date ranges
- Arrays and nested structures for complex metrics

### 4. Interpolation for Gaps

```sql
-- Fill missing time series data points
WITH FILL FROM toDateTime('2024-01-01 00:00:00') 
         TO toDateTime('2024-01-02 00:00:00') 
         STEP 3600
```

---

## ClickHouse for Tracing (Observability)

### OpenTelemetry Integration

ClickHouse has native support for OpenTelemetry, generating trace spans for:
- Query execution
- Query planning
- Distributed query coordination

### Architecture for Trace Storage

```sql
-- Typical traces table schema
CREATE TABLE otel_traces (
    TraceId String,
    SpanId String,
    ParentSpanId String,
    SpanName String,
    ServiceName String,
    StartTime DateTime64(9),
    Duration UInt64,
    StatusCode UInt8,
    Attributes Map(String, String)
) ENGINE = MergeTree()
PARTITION BY toYYYYMMDD(StartTime)
ORDER BY (ServiceName, TraceId, StartTime);
```

### Popular Tracing Backends Using ClickHouse

| Tool | Use Case |
|------|----------|
| **Jaeger** | ClickHouse as storage backend |
| **SigNoz** | Full observability platform built on ClickHouse |
| **Uptrace** | APM and distributed tracing |
| **Grafana Tempo** | ClickHouse as storage option |

---

## ClickHouse for Analytics

### Why It's Fast for Analytics

1. **Vectorized Execution**: Processes data in batches using CPU SIMD instructions
2. **Data Compression**: LZ4/ZSTD compression reduces I/O
3. **Approximate Algorithms**: HyperLogLog, quantiles for fast estimates
4. **Parallel Processing**: Multi-core and distributed query execution

### Example Analytics Query

```sql
-- Real-time analytics dashboard query
SELECT 
    toStartOfMinute(event_time) as minute,
    countIf(event = 'page_view') as page_views,
    countIf(event = 'purchase') as purchases,
    uniqExact(user_id) as unique_users,
    quantile(0.95)(response_time_ms) as p95_latency
FROM events
WHERE event_time >= now() - INTERVAL 1 HOUR
GROUP BY minute
ORDER BY minute;
```

---

## Performance Characteristics

| Metric | ClickHouse Performance |
|--------|----------------------|
| **Insert Speed** | 1-2 million rows/second per server |
| **Query Speed** | Billions of rows scanned per second |
| **Compression** | 10-20x compression typical |
| **Scalability** | Linear horizontal scaling |

---

## When to Use ClickHouse

### Good For ✅

- Log/event analytics
- Time series metrics
- Distributed tracing storage
- Real-time dashboards
- Ad-hoc analytical queries
- IoT data processing

### Not Ideal For ❌

- OLTP (transactional workloads)
- Frequent updates/deletes
- Small datasets (<1GB)
- Point lookups by primary key

---

## Summary

ClickHouse is a powerful analytical database that combines:
- **Column-oriented storage** for compression and fast scans
- **MergeTree engines** for efficient time-based partitioning
- **OpenTelemetry support** for distributed tracing
- **Rich SQL** with time series functions

It's become a go-to choice for observability platforms, replacing traditional time series databases when you need both the performance of specialized TSDBs and the flexibility of SQL analytics.

---

## Resources

- [Official Documentation](https://clickhouse.com/docs)
- [GitHub Repository](https://github.com/ClickHouse/ClickHouse)
- [ClickHouse Blog](https://clickhouse.com/blog)

