# 🧠 local-rag-mini

[![Release](https://img.shields.io/badge/Release-v1.0.0-blue.svg)](https://github.com/MobileConduit/local-rag-mini/releases/tag/v1.0.0)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Privacy: 100% Local](https://img.shields.io/badge/Privacy-100%25%20Local-success.svg)](https://github.com/MobileConduit)
[![Backend: Ollama](https://img.shields.io/badge/Backend-Ollama%20REST-informational.svg)](https://ollama.ai)
[![CI/CD Pipeline](https://github.com/MobileConduit/local-rag-mini/actions/workflows/ci.yml/badge.svg)](https://github.com/MobileConduit/local-rag-mini/actions)

A lightweight, zero-cloud-cost Retrieval-Augmented Generation (RAG) micro-engine designed for enterprise privacy and offline air-gapped environments. Leverages local Ollama embeddings and pure vector similarity math with configurable rate limiting presets and enterprise security headers.

---

## 🏗️ Architecture & Retrieval Flow

The following Mermaid sequence diagram outlines the document chunking, vector indexing, query sanitization, cosine similarity computation, and context retrieval pipeline:

```mermaid
sequenceDiagram
    autonumber
    actor User as User / Client App
    participant API as RAG Gateway (Security & Rate Limiter)
    participant Core as Local RAG Engine
    participant Math as Vector Math Core (Cosine Sim)
    participant Ollama as Local Ollama Instance

    User->>API: Query / API Request (query, top_k)
    API->>API: Sanitize Query & Verify Rate Limits (Strict/Standard/Relaxed)
    alt Rate Limit Exceeded
        API-->>User: 429 Rate Limit Exceeded (Retry-After)
    else Request Approved
        API->>Core: Process Query
        Core->>Ollama: POST /api/embeddings (prompt text)
        Ollama-->>Core: Query Vector Embedding
        loop For Each Document Chunk
            Core->>Math: cosine_similarity(q_vec, chunk_vec)
            Math-->>Core: Normalized Metric [-1.0, 1.0]
        end
        Core->>Core: Sort & Extract Top-K Ranked Contexts
        Core-->>API: Synthesized Search Matches
        API-->>User: 200 OK (Ranked Chunks, Similarity Scores, Security Headers)
    end
```

---

## ⚡ Features

- **Pure Vector Math Engine**: Zero-dependency implementation of Dot Product, Vector Norms, Vector Normalization, Cosine Similarity, and Euclidean Distance.
- **Sliding-Window Rate Limiting**:
  - `strict`: 10 queries/min (edge devices & low-power hardware)
  - `standard`: 30 queries/min (workstation GPU / local Ollama)
  - `relaxed`: 100 queries/min (dedicated enterprise servers)
- **Security & Prompt Defense**: Query sanitization against prompt injection heuristics alongside HTTP security headers (`Content-Security-Policy`, `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Strict-Transport-Security`).
- **Flexible Modes**: Native CLI query runner or lightweight HTTP microservice mode (`--serve`).
- **Zero Cloud Cost**: Complete privacy with offline fallback simulation vectors when Ollama is offline.

---

---

## ⚙️ Installation & Setup

```bash
git clone https://github.com/MobileConduit/local-rag-mini.git
cd local-rag-mini
pip install -r requirements.txt
```

## 🚀 Usage

### 1. Configuration
```bash
cp .env.example .env
```

### 2. Local Ollama Models (Optional for live inference)
```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

### 3. CLI Query Mode
```bash
python rag_engine.py --docs ./company_policies --query "What is the policy regarding remote work and VPN access?" --preset standard
```

### 4. HTTP Microservice Mode
```bash
python rag_engine.py --serve --port 8000 --preset standard
```

Query the API via curl:
```bash
curl -X POST http://localhost:8000 \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the server security policies?", "top_k": 3}'
```

---

## 🧪 Automated Testing

Run the automated test suite verifying vector similarity math, dot products, vector norms, rate limiting, and HTTP security headers:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

---

## 👤 Author & Maintainer

**Eng. MHD. Shadi AL-Hasan**  
- **Role:** Executive CTO & Enterprise Solutions Architect  
- **Email:** [mhd.shadi.alhasan@gmail.com](mailto:mhd.shadi.alhasan@gmail.com)  
- **Phone / WhatsApp:** [+963934005922](tel:+963934005922)  
- **Location:** Damascus, Syria  
- **GitHub:** [MobileConduit](https://github.com/MobileConduit)  

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.  
Copyright (c) 2026 **MHD. Shadi AL-Hasan**. All rights reserved.