# IT Job Finder AI

AI-powered job finder system using FastAPI + RAG + Celery + ChromaDB.

## Project Structure

```bash
it-job-finder-ai/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env
├── main.py
│
├── core/
│   ├── config.py
│   ├── dependencies.py
│   └── logger.py
│
├── routers/
│   ├── chat.py
│   ├── cv.py
│   └── health.py
│
├── services/
│   ├── validation/
│   │   ├── file_validation_service.py
│   │   ├── text_sanitizer_service.py
│   │   └── prompt_injection_service.py
│   │
│   ├── extraction/
│   │   └── extraction_service.py
│   │
│   ├── processing/
│   │   ├── cleaning_service.py
│   │   ├── chunking_service.py
│   │   └── embedding_service.py
│   │
│   ├── storage/
│   │   ├── vector_service.py
│   │   └── metadata_service.py
│   │
│   └── rag/
│       └── rag_service.py
│
├── models/
│   └── schemas.py
│
└── workers/
    ├── celery_app.py
    └── cv_worker.py
```

---

# Architecture Overview

## 1. Infrastructure Layer

### docker-compose.yml

Responsible for orchestrating and managing the AI system infrastructure.

Functions:

- Start containers
- Create internal network
- Mount persistent volumes
- Inject environment variables
- Manage service dependencies

Example:

```yaml
services:
  api:
  redis:
  chromadb:
  celery:
```

---

## 2. Core Layer

Shared system-level components used across the application.

### `config.py`

Centralized application configuration.

Responsibilities:

- Load environment variables
- Store global settings
- Manage secrets/config values

### `dependencies.py`

Dependency Injection layer.

Purpose:

Instead of creating clients repeatedly:

```python
client = chromadb.Client()
```

create reusable dependencies:

```python
def get_vector_db():
    return chromadb.Client()
```

Benefits:

- Cleaner architecture
- Easier testing
- Better maintainability
- Reduced duplicated code

### `logger.py`

Centralized logging configuration.

Responsibilities:

- Structured logs
- Error tracking
- Monitoring support

---

## 3. API Layer (routers)

Equivalent to Express routes in Node.js.

### `chat.py`

Endpoint:

```http
POST /chat
```

Responsibilities:

- Receive user message
- Execute RAG pipeline
- Return AI response

---

### `cv.py`

Endpoint:

```http
POST /upload-cv
```

Responsibilities:

- Upload CV file
- Validate input
- Enqueue Celery task

---

### `health.py`

Endpoint:

```http
GET /health
```

Response:

```json
{
    "status":"ok"
}
```

Used by:

- Docker health checks
- Kubernetes probes
- Monitoring systems

---

## 4. Service Layer

Contains business logic of the application.

### Validation Services

Responsible for security and input validation.

**file_validation_service.py**

- Validate file type
- Validate size

**text_sanitizer_service.py**

- Clean text
- Remove unwanted characters

**prompt_injection_service.py**

- Detect malicious prompts
- Prevent prompt injection attacks

---

### Extraction Services

**extraction_service.py**

Responsibilities:

- Extract text from CV
- Parse structured information

Example:

- Name
- Skills
- Experience
- Education

---

### Processing Services

**cleaning_service.py**

- Normalize text

**chunking_service.py**

- Split large text into chunks

**embedding_service.py**

- Convert text into vectors

---

### Storage Services

**vector_service.py**

Responsibilities:

- Store embeddings
- Query vector database

**metadata_service.py**

Responsibilities:

- Store structured metadata

---

### RAG Service

**rag_service.py**

Main retrieval pipeline:

```text
User Query
    ↓
Embedding
    ↓
Vector Search
    ↓
Retrieve Context
    ↓
LLM Generation
    ↓
Response
```

---

## 5. Worker Layer

### `celery_app.py`

Celery configuration.

Responsibilities:

- Queue settings
- Broker connection
- Worker configuration

---

### `cv_worker.py`

Background processing task.

Responsibilities:

- CV extraction
- Chunking
- Embedding generation
- Vector storage

---

# Runtime Flow

Actual workflow when a user uploads a CV:

```text
Frontend
   ↓
FastAPI Upload Endpoint
   ↓
Enqueue Task
   ↓
Redis Queue
   ↓
Celery Worker
   ↓
Extract Text
   ↓
Cleaning
   ↓
Chunking
   ↓
Embedding
   ↓
Store Vector Database
```

---

# Tech Stack

- FastAPI
- ChromaDB
- Redis
- Celery
- Docker
- RAG
- Embedding Models
- Python

---

# Design Principles

- Clean Architecture
- Separation of Concerns
- Dependency Injection
- Async Processing
- Scalable Infrastructure
- Production-ready structure
