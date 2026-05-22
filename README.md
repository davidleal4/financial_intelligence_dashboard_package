# 📊 Financial Intelligence Dashboard

An agentic Retrieval-Augmented Generation (RAG) dashboard that bridges the gap between static financial filings and live market data. 

This project demonstrates a production-grade approach to building AI analysts. It uses a ReAct (Reasoning and Acting) agent to intelligently route queries between a local vector database containing complex document contexts (like 10-Ks) and real-time market APIs.


## ⚡ Key Features

* **Intelligent Tool Routing (ReAct Agent):** The core LLM doesn't just guess; it plans. It decides whether to query the document vector store, fetch live market data, or combine both to answer complex comparative questions.
* **Advanced Document Parsing:** Uses **Sentence-Window Retrieval** to index financial PDFs. Instead of chunking blindly, it retrieves the exact semantic sentence and injects the surrounding window of text, preserving critical financial context and preventing hallucination.
* **Live Market Intelligence:** Integrates with `yfinance` to pull real-time pricing, P/E ratios, market caps, historical charts, and recent news.
* **Auditability & Traceability:** Enterprise-grade AI requires trust. The dashboard exposes the agent's internal "Thought Trace" (actions and tool inputs) and provides strict, UI-linked source citations with similarity scores for every document claim.
* **Local Vector Storage:** Uses ChromaDB for persistent, fast, and local vector embeddings.

## 🛠️ Architecture & Tech Stack

* **Frontend:** Streamlit (Responsive UI, session state management, async execution bridging)
* **Orchestration:** LlamaIndex (Agent workflows, node parsing, tool creation)
* **LLM & Embeddings:** OpenAI (`gpt-4o-mini`, `text-embedding-3-small`)
* **Vector Database:** ChromaDB (Persistent local storage)
* **Data Sources:** PDF Documents (via `SimpleDirectoryReader`) + Live Market Data (via `yfinance` and `pandas`)

## 🚀 Getting Started

### Prerequisites
* Python 3.9+
* An OpenAI API Key

### Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/financial-intelligence-dashboard.git](https://github.com/yourusername/financial-intelligence-dashboard.git)
   cd financial-intelligence-dashboard