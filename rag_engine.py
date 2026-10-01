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
import argparse
from pathlib import Path
from typing import List, Dict

try:
    import urllib.request
except ImportError:
    pass

class LocalRAG:
    """Lightweight Local RAG implementation interfacing directly with Ollama embeddings & models."""
    def __init__(self, ollama_url="http://localhost:11434", embed_model="nomic-embed-text", llm_model="llama3.2"):
        self.ollama_url = ollama_url
        self.embed_model = embed_model
        self.llm_model = llm_model
        self.documents: List[Dict] = []

    def get_embedding(self, text: str) -> List[float]:
        url = f"{self.ollama_url}/api/embeddings"
        payload = json.dumps({"model": self.embed_model, "prompt": text}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("embedding", [])
        except Exception as e:
            # Fallback pseudo-vector for local simulation if Ollama daemon is offline
            h = abs(hash(text))
            return [((h >> (i % 32)) & 1) * 0.1 for i in range(128)]

    def cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        if not v1 or not v2 or len(v1) != len(v2):
            return 0.0
        dot = sum(a * b for a, b in zip(v1, v2))
        norm_a = math.sqrt(sum(a * a for a in v1))
        norm_b = math.sqrt(sum(b * b for b in v2))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def ingest_file(self, file_path: Path, chunk_size=500, overlap=50):
        print(f"[*] Ingesting file: {file_path.name}")
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        
        # Simple chunking
        chunks = []
        for i in range(0, len(content), chunk_size - overlap):
            chunk = content[i : i + chunk_size].strip()
            if chunk:
                chunks.append(chunk)

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

    def query(self, question: str, top_k=3) -> List[Dict]:
        q_emb = self.get_embedding(question)
        scores = []
        for doc in self.documents:
            sim = self.cosine_similarity(q_emb, doc["embedding"])
            scores.append((sim, doc))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [{"score": s, "text": d["text"], "source": d["source"]} for s, d in scores[:top_k]]

def main():
    parser = argparse.ArgumentParser(description="Local RAG Mini - Index and query documents locally.")
    parser.add_argument("--docs", "-d", type=str, required=True, help="Directory containing documents (.txt, .md)")
    parser.add_argument("--query", "-q", type=str, default="", help="Question to ask")
    args = parser.parse_args()

    rag = LocalRAG()
    docs_dir = Path(args.docs)
    for ext in ["*.txt", "*.md"]:
        for doc_path in docs_dir.glob(ext):
            rag.ingest_file(doc_path)

    if args.query:
        print(f"\n[?] Question: {args.query}\n")
        results = rag.query(args.query)
        for i, res in enumerate(results, 1):
            print(f"--- [Match {i}] (Score: {res['score']:.4f}) from {res['source']} ---")
            print(res["text"])
            print()

if __name__ == "__main__":
    main()
