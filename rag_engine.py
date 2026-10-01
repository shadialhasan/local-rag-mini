# -*- coding: utf-8 -*-
"""
Local RAG Mini - Zero Cloud Cost Document Intelligence Engine
Author: Eng. MHD. Shadi AL-Hasan <mhd.shadi.alhasan@gmail.com>
Phone: +963934005922
Copyright (c) 2026 MHD. Shadi AL-Hasan
"""

import os
import sys
import json
import math
import time
import re
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from http.server import HTTPServer, BaseHTTPRequestHandler

try:
    import urllib.request
except ImportError:
    pass

# Rate Limiting Presets (QPM: Queries Per Minute)
RATE_LIMIT_PRESETS = {
    "strict": {"qpm": 10, "burst": 2},
    "standard": {"qpm": 30, "burst": 5},
    "relaxed": {"qpm": 100, "burst": 15},
}

# Standard Security Headers for HTTP API Mode
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
    "Cache-Control": "no-store, no-cache, must-revalidate",
}


# ==============================================================================
# Vector Similarity Math Engine
# ==============================================================================

def dot_product(v1: List[float], v2: List[float]) -> float:
    """Computes the dot product between two real-valued vectors."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    return sum(a * b for a, b in zip(v1, v2))


def vector_norm(v: List[float]) -> float:
    """Computes the Euclidean L2 norm of a vector."""
    if not v:
        return 0.0
    return math.sqrt(sum(x * x for x in v))


def normalize_vector(v: List[float]) -> List[float]:
    """Normalizes a vector to unit length (L2 norm = 1.0)."""
    norm = vector_norm(v)
    if norm == 0.0:
        return [0.0] * len(v)
    return [x / norm for x in v]


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """
    Computes cosine similarity between two vectors in [-1.0, 1.0].
    Returns 0.0 if either vector is empty, zero-norm, or dimensions mismatch.
    """
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = dot_product(v1, v2)
    norm_a = vector_norm(v1)
    norm_b = vector_norm(v2)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    # Guard against floating point rounding outside [-1.0, 1.0]
    return max(-1.0, min(1.0, dot / (norm_a * norm_b)))


def euclidean_distance(v1: List[float], v2: List[float]) -> float:
    """Computes Euclidean distance between two vectors."""
    if not v1 or not v2 or len(v1) != len(v2):
        return float("inf")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))


def sanitize_query(query: str) -> Tuple[str, bool]:
    """
    Sanitizes user query and detects basic prompt injection patterns.
    Returns (cleaned_query, is_suspicious).
    """
    if not query:
        return "", False
    cleaned = query.strip()
    suspicious_patterns = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"system\s+prompt\s+override",
        r"you\s+are\s+now\s+in\s+dan\s+mode",
        r"reveal\s+(your\s+)?secret\s+key"
    ]
    is_suspicious = any(re.search(pat, cleaned, re.IGNORECASE) for pat in suspicious_patterns)
    return cleaned, is_suspicious


# ==============================================================================
# Rate Limiting Engine
# ==============================================================================

class RAGRateLimiter:
    """Sliding-window rate limiter designed for edge RAG inference protection."""

    def __init__(self, preset: str = "standard", window_seconds: float = 60.0):
        preset_config = RATE_LIMIT_PRESETS.get(preset, RATE_LIMIT_PRESETS["standard"])
        self.qpm_limit = preset_config["qpm"]
        self.window_seconds = window_seconds
        self.client_history: Dict[str, List[float]] = {}

    def check_limit(self, client_id: str = "default") -> Tuple[bool, str, dict]:
        now = time.time()
        cutoff = now - self.window_seconds
        timestamps = self.client_history.setdefault(client_id, [])
        self.client_history[client_id] = [t for t in timestamps if t > cutoff]

        current_count = len(self.client_history[client_id])
        if current_count >= self.qpm_limit:
            oldest = self.client_history[client_id][0]
            retry_after = max(1, int(self.window_seconds - (now - oldest)))
            return False, f"Rate limit exceeded: max {self.qpm_limit} QPM", {
                "limit_qpm": self.qpm_limit,
                "current_qpm": current_count,
                "retry_after": retry_after
            }

        self.client_history[client_id].append(now)
        return True, "", {
            "limit_qpm": self.qpm_limit,
            "remaining_qpm": self.qpm_limit - (current_count + 1)
        }


# ==============================================================================
# Local RAG Core Engine
# ==============================================================================

class LocalRAG:
    """Lightweight Local RAG implementation interfacing directly with Ollama embeddings & models."""

    _ollama_offline: bool = False

    def __init__(
        self,
        ollama_url: str = None,
        embed_model: str = None,
        llm_model: str = None,
        rate_limit_preset: str = "standard"
    ):
        self.ollama_url = ollama_url or os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.embed_model = embed_model or os.getenv("EMBED_MODEL", "nomic-embed-text")
        self.llm_model = llm_model or os.getenv("LLM_MODEL", "llama3.2")
        self.rate_limiter = RAGRateLimiter(preset=rate_limit_preset or os.getenv("RATE_LIMIT_PRESET", "standard"))
        self.documents: List[Dict] = []

    def get_embedding(self, text: str) -> List[float]:
        if not LocalRAG._ollama_offline:
            url = f"{self.ollama_url}/api/embeddings"
            payload = json.dumps({"model": self.embed_model, "prompt": text}).encode("utf-8")
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=0.2) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data.get("embedding", [])
            except Exception:
                LocalRAG._ollama_offline = True

        # Deterministic fallback pseudo-vector for offline simulation & testing
        h = abs(hash(text))
        vec = [((h >> (i % 32)) & 1) * 0.1 for i in range(128)]
        return normalize_vector(vec)

    def cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Computes cosine similarity delegating to core vector math engine."""
        return cosine_similarity(v1, v2)

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Splits raw text into overlapping contiguous chunks."""
        if not text:
            return []
        step = max(1, chunk_size - overlap)
        chunks = []
        for i in range(0, len(text), step):
            chunk = text[i : i + chunk_size].strip()
            if chunk:
                chunks.append(chunk)
        return chunks

    def ingest_file(self, file_path: Path, chunk_size: int = 500, overlap: int = 50):
        print(f"[*] Ingesting file: {file_path.name}")
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        chunks = self.chunk_text(content, chunk_size, overlap)

        print(f"[*] Created {len(chunks)} text chunks. Generating embeddings...")
        for idx, chunk in enumerate(chunks):
            emb = self.get_embedding(chunk)
            self.documents.append({
                "source": file_path.name,
                "chunk_id": idx,
                "text": chunk,
                "embedding": emb
            })
        print(f"[+] Successfully indexed {len(chunks)} chunks from {file_path.name}")

    def ingest_text(self, text: str, source_label: str = "memory", chunk_size: int = 500, overlap: int = 50):
        """Direct text ingestion into memory store."""
        chunks = self.chunk_text(text, chunk_size, overlap)
        for idx, chunk in enumerate(chunks):
            emb = self.get_embedding(chunk)
            self.documents.append({
                "source": source_label,
                "chunk_id": idx,
                "text": chunk,
                "embedding": emb
            })

    def query(self, question: str, top_k: int = 3, client_id: str = "default") -> List[Dict]:
        """Queries the indexed knowledge store with rate limiting and vector scoring."""
        allowed, msg, meta = self.rate_limiter.check_limit(client_id)
        if not allowed:
            raise PermissionError(f"{msg}. Retry after {meta.get('retry_after', 1)}s")

        sanitized, suspicious = sanitize_query(question)
        q_emb = self.get_embedding(sanitized)

        scores = []
        for doc in self.documents:
            sim = cosine_similarity(q_emb, doc["embedding"])
            scores.append((sim, doc))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "score": round(float(s), 4),
                "text": d["text"],
                "source": d["source"],
                "chunk_id": d.get("chunk_id", 0),
                "flagged_suspicious": suspicious
            }
            for s, d in scores[:top_k]
        ]


# ==============================================================================
# Optional HTTP Server Mode
# ==============================================================================

def create_rag_http_handler(rag_instance: LocalRAG):
    class RAGHTTPHandler(BaseHTTPRequestHandler):
        def apply_security_headers(self):
            for k, v in SECURITY_HEADERS.items():
                self.send_header(k, v)

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.apply_security_headers()
            self.end_headers()
            self.wfile.write(json.dumps({
                "service": "local-rag-mini",
                "indexed_documents": len(rag_instance.documents),
                "embed_model": rag_instance.embed_model,
                "llm_model": rag_instance.llm_model,
                "author": "Eng. MHD. Shadi AL-Hasan"
            }).encode("utf-8"))

        def do_POST(self):
            client_ip = self.client_address[0] if self.client_address else "127.0.0.1"
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"

            try:
                data = json.loads(body)
                query_text = data.get("query", "")
                top_k = int(data.get("top_k", 3))

                results = rag_instance.query(query_text, top_k=top_k, client_id=client_ip)

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.apply_security_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"results": results}).encode("utf-8"))

            except PermissionError as pe:
                self.send_response(429)
                self.send_header("Content-Type", "application/json")
                self.apply_security_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(pe)}).encode("utf-8"))
            except Exception as e:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.apply_security_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

    return RAGHTTPHandler


def main():
    parser = argparse.ArgumentParser(description="Local RAG Mini - Index and query documents locally.")
    parser.add_argument("--docs", "-d", type=str, default="", help="Directory containing documents (.txt, .md)")
    parser.add_argument("--query", "-q", type=str, default="", help="Question to ask")
    parser.add_argument("--top_k", "-k", type=int, default=3, help="Number of retrieved chunks")
    parser.add_argument("--serve", action="store_true", help="Launch HTTP microservice")
    parser.add_argument("--port", "-p", type=int, default=8000, help="HTTP server port")
    parser.add_argument("--preset", choices=["strict", "standard", "relaxed"], default="standard", help="Rate limit preset")
    args = parser.parse_args()

    rag = LocalRAG(rate_limit_preset=args.preset)

    if args.docs:
        docs_dir = Path(args.docs)
        if docs_dir.exists() and docs_dir.is_dir():
            for ext in ["*.txt", "*.md"]:
                for doc_path in docs_dir.glob(ext):
                    rag.ingest_file(doc_path)
        else:
            print(f"[!] Warning: Provided docs directory '{args.docs}' does not exist.")

    if args.serve:
        handler = create_rag_http_handler(rag)
        server = HTTPServer(("0.0.0.0", args.port), handler)
        print(f"[*] Local RAG Mini HTTP service active on port {args.port} [Preset: {args.preset}]...")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n[!] Stopping server.")
        return

    if args.query:
        print(f"\n[?] Question: {args.query}\n")
        results = rag.query(args.query, top_k=args.top_k)
        for i, res in enumerate(results, 1):
            print(f"--- [Match {i}] (Score: {res['score']:.4f}) from {res['source']} ---")
            print(res["text"])
            print()


if __name__ == "__main__":
    main()
