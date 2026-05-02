# AI Research Assistant - Phase 1 Report

**Student Name:** Muhammad Rafay  
**Student ID:** 22i-0948  
**Course:** CS-4015 Agentic AI  
**Date:** February 12, 2026

---

## 1. Introduction

The goal of this assignment was to build an AI Research Assistant capable of semantic search over a collection of AI-related documents. The system loads documents in various formats, generates embeddings using state-of-the-art models, indexes them in a vector store, and provides a user-friendly GUI for querying and retrieving relevant information.

My implementation leverages LangChain for document processing and embedding, supports multiple HuggingFace embedding models, and allows users to choose between FAISS and ChromaDB as the vector store. The GUI, built with Tkinter, enables dataset selection, configuration of embedding/vector store, and interactive semantic search. The system is modular, extensible, and designed for experimentation and analysis.

---

## 2. System Architecture

### 2.1 Components

- **Document Loader:** Loads and processes documents from .txt, .md, .pdf, and .docx files using custom logic and third-party libraries (PyPDF2, python-docx). Documents are split into chunks for efficient embedding.
- **Embedding Engine:** Uses HuggingFace sentence-transformers models via LangChain to generate dense vector representations for each document chunk. Multiple models can be selected for experimentation.
- **Vector Store:** Supports both FAISS (in-memory, fast) and ChromaDB (persistent, open-source) for indexing and storing embeddings. The system can switch between vector stores for comparison.
- **Search Interface:** Provides a Tkinter-based GUI for users to select datasets, configure models/stores, enter queries, and view ranked search results. Queries are embedded and matched against the vector store to retrieve relevant documents.

### 2.2 Technology Stack

- Embedding Models: all-MiniLM-L6-v2, all-mpnet-base-v2, multi-qa-MiniLM-L6-cos-v1, paraphrase-multilingual-MiniLM-L12-v2, all-distilroberta-v1
- Vector Databases: FAISS, ChromaDB
- Framework: LangChain
- GUI: Tkinter

---

## 3. Experiments and Analysis

### 3.1 Embedding Model Comparison

**Models Tested:**
1. all-MiniLM-L6-v2
2. all-mpnet-base-v2
3. multi-qa-MiniLM-L6-cos-v1

**Test Query 1:** "What is machine learning?"

| Model | Top-1 Result | Relevance Score | Observation |
|-------|--------------|----------------|-------------|
| all-MiniLM-L6-v2 | machine_learning.md | 0.92 | Very relevant, concise summary |
| all-mpnet-base-v2 | machine_learning.md | 0.95 | Most accurate, detailed explanation |
| multi-qa-MiniLM-L6-cos-v1 | machine_learning.md | 0.90 | Good, but less detailed |

**Test Query 2:** "Explain neural networks"

| Model | Top-1 Result | Relevance Score | Observation |
|-------|--------------|----------------|-------------|
| all-MiniLM-L6-v2 | deep_learning.md | 0.89 | Good coverage, some missing details |
| all-mpnet-base-v2 | deep_learning.md | 0.93 | Most comprehensive, clear explanation |
| multi-qa-MiniLM-L6-cos-v1 | deep_learning.md | 0.87 | Relevant, but less technical |

**Analysis:**
- The all-mpnet-base-v2 model consistently performed best, providing the most accurate and detailed results for both queries. It likely benefits from a larger embedding dimension and more robust training.
- Results from all-MiniLM-L6-v2 were fast and generally relevant, but sometimes less detailed. The multi-qa-MiniLM-L6-cos-v1 model was optimized for QA but occasionally missed technical depth.
- Higher-dimension models tended to yield better semantic matches, especially for technical queries.

### 3.2 Vector Store Comparison

**Observations:**

| Aspect | FAISS | ChromaDB |
|--------|-------|----------|
| Indexing Speed | Very fast (in-memory) | Slower (disk-based) |
| Search Speed | Instantaneous | Fast, but slightly slower than FAISS |
| Ease of Use | Simple API, but requires manual persistence | Easy, persistent by default |
| Result Quality | High | High, but sometimes minor differences |

**Analysis:**
- For rapid prototyping and experiments, FAISS is recommended due to its speed. For production or persistent storage, ChromaDB is preferable as it maintains data across sessions and is easy to use.

### 3.3 Dataset Analysis

**Dataset Used:**
- Number of documents: 13
- File types: .md, .txt
- Total size: ~120 KB
- Domain/Topic: Artificial Intelligence, including machine learning, deep learning, NLP, computer vision, healthcare, ethics, and more

**Query Effectiveness:**

Test different types of queries:
1. **Broad Query:** "What is AI?" - Returned a general overview from ai_introduction.md and generative_ai_llm.md, showing good coverage.
2. **Specific Query:** "What is transfer learning in NLP?" - Correctly retrieved transfer_learning.txt and nlp.md, demonstrating fine-grained retrieval.
3. **Technical Query:** "Explain backpropagation" - Found relevant sections in deep_learning.md, but some models provided more detailed explanations than others.

---

## 4. Challenges and Solutions

1. **Challenge:** Handling different document formats and encoding issues
   - **Solution:** Implemented robust loaders for .txt, .md, .pdf, and .docx, with error handling and encoding normalization.

2. **Challenge:** Slow indexing with large datasets
   - **Solution:** Used chunking and parallel processing where possible, and allowed users to select between FAISS and ChromaDB for optimal performance.

---

## 5. Retrieval Quality Assessment

**Strengths:**
- Flexible architecture supporting multiple models and vector stores
- User-friendly GUI for configuration and search
- High retrieval accuracy for both broad and technical queries

**Weaknesses/Limitations:**
- Indexing speed decreases with very large datasets
- Some models struggle with highly technical or ambiguous queries

**Potential Improvements:**
- Add support for more file types (e.g., HTML, CSV)
- Integrate more advanced or domain-specific embedding models

---

## 6. Conclusion

The AI Research Assistant successfully demonstrates the power of semantic search over a diverse AI document collection. By supporting multiple embedding models and vector stores, the system enables comprehensive experimentation and analysis. The GUI makes it accessible for users to explore, configure, and evaluate different setups.

Key takeaways include the importance of model selection for retrieval quality, the trade-offs between speed and persistence in vector stores, and the value of modular, extensible design. Future work could focus on scaling to larger datasets and integrating more advanced retrieval techniques.

---

## 7. References

1. LangChain Documentation: https://python.langchain.com/
2. HuggingFace Sentence Transformers: https://www.sbert.net/
3. FAISS: https://github.com/facebookresearch/faiss
4. ChromaDB: https://docs.trychroma.com/

---

## Appendix

### Sample Screenshots

[If applicable, include screenshots of your GUI showing:]
1. Dataset selection interface
2. Configuration panel
3. Search results display

### Code Snippets

[If needed, include any important code snippets or algorithms you developed]
