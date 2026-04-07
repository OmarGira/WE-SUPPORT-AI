# 💜 WE Support AI - Enterprise Assistant

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B.svg?logo=streamlit&logoColor=white)
![AI Model](https://img.shields.io/badge/AI-Gemini_2.5_Flash-orange.svg)
![Vector DB](https://img.shields.io/badge/Vector_DB-FAISS_%7C_BM25-yellow.svg)

A production-grade, microservices-based AI Assistant tailored for Telecom Egypt (WE). This project leverages Advanced Retrieval-Augmented Generation (Hybrid RAG) and Multimodal Vision to provide highly accurate, context-aware customer support.

---

## 📑 Table of Contents
1. [System Architecture](#-system-architecture)
2. [Key Features](#-key-features)
3. [Tech Stack](#-tech-stack)
4. [Prerequisites & Setup](#-prerequisites--setup)
5. [Data Ingestion Pipeline](#-data-ingestion-pipeline-important)
6. [How to Run](#-how-to-run)
7. [API Documentation](#-api-documentation)

---

## 🏗️ System Architecture
The application is decoupled into distinct microservices for scalability:
- **Frontend (Streamlit):** A "dumb UI" responsible only for user interaction, file/HTML parsing, and rendering markdown/images.
- **Backend (FastAPI):** A robust REST API that handles inference requests, rate-limiting, and error handling.
- **AI Core Engine:** - **Hybrid Retriever:** Combines `FAISS` (Dense/Semantic Search via `MiniLM-L12-v2`) and `BM25` (Sparse/Keyword Search) for zero-hallucination data retrieval.
  - **Generator:** Google GenAI (`Gemini 2.5 Flash`) for text and multimodal generation.
  - **Local Vision Model:** A Lazy-loaded `BLIP` model for local image captioning to save cloud API tokens.

---

## 🌟 Key Features
- **Hybrid RAG System:** Alpha-blended scoring (60% Dense / 40% Sparse) to ensure exact matches for error codes and semantic understanding for natural language.
- **Multimodal Capabilities:** Upload images (e.g., router lights, error screens). Choose between High-Fidelity Cloud Vision (Gemini) or Zero-Token Local Vision (BLIP).
- **Omni-Document Parsing:** Supports uploading PDF, DOCX, TXT, and extracting clean text from raw HTML codes using `BeautifulSoup`.
- **Intelligent Web Scraper:** Asynchronous crawler using `Playwright` to extract dynamic JS-rendered pricing tables from the official TE website.
- **Rate-Limit Resilience:** Exponential backoff mechanism to bypass `429 Too Many Requests` from cloud APIs.

---

## 🛠️ Tech Stack
- **Web Scraping:** Playwright (Async), BeautifulSoup4
- **Data Engineering:** LangChain (RecursiveCharacterTextSplitter)
- **AI & NLP:** Google GenAI SDK, HuggingFace Transformers, Sentence-Transformers
- **Backend:** FastAPI, Uvicorn, Pydantic
- **Frontend:** Streamlit, Pillow, PyPDF2, python-docx

---

## ⚙️ Prerequisites & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/OmarGira/WE-Support-AI.git](https://github.com/OmarGira/WE-Support-AI.git)
cd WE-Support-AI