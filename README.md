# 🧠 local-rag-mini

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Privacy: 100% Local](https://img.shields.io/badge/Privacy-100%25%20Local-success.svg)](https://github.com/MobileConduit)
[![Backend: Ollama](https://img.shields.io/badge/Backend-Ollama%20REST-informational.svg)](https://ollama.ai)

A lightweight, zero-cloud-cost Retrieval-Augmented Generation (RAG) micro-engine. Designed for enterprise privacy and offline environments using local Ollama embeddings and pure vector similarity without bulky frameworks.

---

## 💡 Why This Project?
Most modern RAG tutorials mandate heavy dependencies (LangChain, LlamaIndex, massive C++ vector drivers). `local-rag-mini` is built to run on standard edge servers and internal networks with **zero external telemetry and zero per-token cost**.

## 🚀 Usage

1. Start your local Ollama server:
```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

2. Run query over your private documents folder:
```bash
python rag_engine.py --docs ./company_policies --query "What is the policy regarding remote work and VPN access?"
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

