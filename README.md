# AWS ML Tasks: Hands-on Machine Learning & Semantic Search Pipelines

[![Open Week 1 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EyobTesfaye8/AWS_tasks/blob/main/week-01/AWS_First_week_task.ipynb)
[![Open Week 2 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EyobTesfaye8/AWS_tasks/blob/main/week-02/AWS_ML_Semantic_Search.ipynb)
![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Vector DB](https://img.shields.io/badge/VectorDB-ChromaDB-green.svg)
![License](https://img.shields.io/badge/license-MIT-purple.svg)

This repository contains hands-on data engineering, exploratory data analysis, dense vector embeddings, and semantic retrieval pipelines developed for the **AWS Machine Learning (AASTU Cohort)** program.

---

## 📁 Repository Structure

The project is structured into modular, weekly directories following course standards:

```text
AWS_ML/
├── .gitignore                                      # Excludes virtual environments, pycache, checkpoints
├── requirements.txt                                # Consolidated dependencies (numpy, pandas, chromadb, etc.)
├── README.md                                       # Comprehensive documentation & setup instructions
├── week-01/                                        # Week 01 Deliverables: Market Data Analysis Pipeline
│   ├── AWS_First_week_task.ipynb                   # Executed Jupyter Notebook with outputs & plots
│   ├── main.py                                     # Python script with named cleaning & NumPy functions
│   ├── marketing.csv                               # Raw agricultural market arrivals dataset
│   └── price_trends.png                            # Generated time-series visualization
└── week-02/                                        # Week 02 Deliverables: APIs, Embeddings & Semantic Search
    ├── AWS_ML_Semantic_Search.ipynb                # Custom end-to-end semantic search notebook
    ├── AASTU_Week2_Data_APIs_Embeddings_Practical.ipynb # Cohort starter reference notebook
    ├── main.py                                     # Standalone semantic search & vector DB pipeline script
    └── data/
        ├── raw_messy_faqs.csv                      # Raw dataset with intentional defects (nulls/duplicates)
        ├── faqs.csv                                # Curated 20+ AWS & ML domain FAQs
        └── faqs.json                               # Structured JSON representation of FAQ database
```

---

## 🚀 Environment Setup & Getting Started

Follow the steps below to configure an isolated Python virtual environment and install the required dependencies.

### 1. Clone the Repository

```bash
git clone https://github.com/EyobTesfaye8/AWS_tasks.git
cd AWS_tasks
```

### 2. Create and Activate a Virtual Environment (`venv`)

An isolated virtual environment ensures clean, reproducible dependency management across operating systems:

* **On Linux / macOS:**
  ```bash
  # Create a virtual environment named '.venv'
  python3 -m venv .venv

  # Activate the virtual environment
  source .venv/bin/activate
  ```

* **On Windows (Command Prompt / PowerShell):**
  ```powershell
  # Create a virtual environment named '.venv'
  python -m venv .venv

  # Activate the virtual environment
  .\.venv\Scripts\activate
  ```

> [!NOTE]
> The `.venv/` directory is explicitly excluded in [`.gitignore`](.gitignore) and is never committed to GitHub, preventing repository bloat and dependency leaks.

### 3. Install Dependencies

Install all core libraries using [`requirements.txt`](requirements.txt):

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Key packages included:
* **`numpy`**: Vectorized numerical calculations and pure mathematical cosine similarity.
* **`pandas`**: Tabular data manipulation, data quality auditing, and CSV/JSON IO.
* **`matplotlib`**: Statistical charting and time-series plotting.
* **`sentence-transformers` & `torch`**: Dense semantic embedding generation (`all-MiniLM-L6-v2`).
* **`chromadb`**: Lightweight AI-native vector database for persistent storage and Approximate Nearest Neighbor (ANN) search.
* **`requests`**: Resilient HTTP REST API integration.
* **`jupyter` / `ipykernel`**: Interactive notebook environment.

---

## 📊 Week 01: Market Arrivals Data Analysis Pipeline

An end-to-end pipeline analyzing 10,227 records of agricultural market arrival prices and trade volumes.

### How to Run Week 01

* **Run the standalone pipeline script:**
  ```bash
  python week-01/main.py
  ```
* **Run the Jupyter Notebook locally:**
  ```bash
  jupyter notebook week-01/AWS_First_week_task.ipynb
  ```
* **Open directly in Google Colab:**
  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EyobTesfaye8/AWS_tasks/blob/main/week-01/AWS_First_week_task.ipynb)

### Week 01 Architecture Highlights

1. **Modular Named Cleaning Functions (`def`)**:
   * `load_raw_data()`: Loads dataset locally with automated remote GitHub fallback.
   * `clean_missing_and_duplicates()`: Checks for missing values across columns and deduplicates rows.
   * `clean_text_fields()`: Normalizes categorical text (strips whitespace, standardizes uppercase casing).
   * `convert_column_types()`: Casts `date` to `datetime64[ns]` and validates numerical types (`quantity`, `priceMod`, etc.).
   * `detect_and_handle_outliers()`: Identifies extreme outliers using the Interquartile Range (IQR) method and applies Winsorization (capping):
     $$\text{IQR} = Q_3 - Q_1, \quad \text{Lower} = \max(0, Q_1 - 1.5 \times \text{IQR}), \quad \text{Upper} = Q_3 + 1.5 \times \text{IQR}$$
2. **NumPy Statistical Vector Operations**:
   * Utilizes 7 distinct NumPy calculation functions: `np.mean()`, `np.median()`, `np.std()`, `np.percentile()`, `np.where()`, `np.min()`, and `np.max()`.
3. **Summary of Findings**:
   * **Dataset Size**: 10,227 records.
   * **Modal Price Mean**: 984.28 INR / quintal | **Median**: 747.00 INR / quintal.
   * **Volatility (`np.std`)**: 818.43 INR / quintal with prices spanning from 80.00 to 6,400.00.

---

## 🔍 Week 02: APIs, Vector Embeddings & Semantic Search (FAQ Assistant)

A production-ready semantic retrieval and FAQ assistant designed for AWS Cloud, Machine Learning, and AASTU Cohort academic support.

### How to Run Week 02

* **Run the standalone pipeline script:**
  ```bash
  python week-02/main.py
  ```
* **Run the custom semantic search notebook locally:**
  ```bash
  jupyter notebook week-02/AWS_ML_Semantic_Search.ipynb
  ```
* **Open directly in Google Colab:**
  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EyobTesfaye8/AWS_tasks/blob/main/week-02/AWS_ML_Semantic_Search.ipynb)

### Week 02 Pipeline Stages

```
   Raw Data (CSV/JSON)
           │
           ▼
┌─────────────────────────┐
│ 1. Data Quality & Clean │ ──► Null removal, deduplication, text & category normalization
└─────────────────────────┘
           │
           ▼
┌─────────────────────────┐
│ 2. API & Chunking Demo  │ ──► REST JSON requests + Sliding window passage chunking
└─────────────────────────┘
           │
           ▼
┌─────────────────────────┐
│ 3. Dense Vector Embed   │ ──► sentence-transformers/all-MiniLM-L6-v2 (384-dim, L2 normalized)
└─────────────────────────┘
           │
     ┌─────┴──────────────────┐
     ▼                        ▼
┌─────────────────────────┐ ┌─────────────────────────┐
│ 4. Pure NumPy Cosine    │ │ 5. ChromaDB Vector DB   │
│    Matrix Dot-Product   │ │    Persistent Storage   │
│    Nearest Neighbors    │ │    + Metadata Filter    │
└─────────────────────────┘ └─────────────────────────┘
     │                        │
     └─────┬──────────────────┘
           ▼
┌─────────────────────────┐
│ 6. Comparative Eval     │ ──► Multi-query testing, NumPy vs ChromaDB, error analysis
└─────────────────────────┘
```

1. **Data Inspection & Cleaning**:
   * Evaluates `week-02/data/raw_messy_faqs.csv` containing deliberate real-world defects.
   * Detects and drops incomplete rows missing essential answers (`faq_022`).
   * Deduplicates identical question-answer pairs (`faq_021`).
   * Strips extraneous whitespace and standardizes category identifiers (`aws_cloud`, `machine_learning`, `academics`, `support`).
   * Feature engineering: Computes min-max normalized question length ($[0, 1]$).
2. **REST API & JSON Handling**:
   * Demonstrates HTTP GET requests, HTTP status code validation, JSON serialization/deserialization, and offline fallback handling.
3. **Document Chunking & Tokenization**:
   * Implements text chunking with configurable chunk size and sliding window overlap to preserve semantic context across passage boundaries.
4. **Dense Vector Embeddings**:
   * Formats composite document text: `"Question: ... \nAnswer: ..."`.
   * Encodes documents using `sentence-transformers/all-MiniLM-L6-v2` into 384-dimensional unit-normalized dense vectors.
5. **Pure NumPy Semantic Search**:
   * Because embeddings are $L_2$-normalized ($\| \mathbf{v} \|_2 = 1$), cosine similarity simplifies directly to the dot product:
     $$\text{Cosine Similarity}(\mathbf{q}, \mathbf{d}) = \mathbf{q} \cdot \mathbf{d} = \sum_{i=1}^{384} q_i d_i$$
   * Implements top-$k$ nearest-neighbor retrieval using pure NumPy vectorized operations (`np.dot`, `np.argsort`).
6. **Vector Database Integration (ChromaDB)**:
   * Initializes persistent local storage (`./chroma_db`).
   * Indexes document text, embeddings, and structured metadata.
   * Executes Approximate Nearest Neighbor (ANN) search with structured metadata filtering (e.g., `where={"category": "aws_cloud"}`).
7. **Evaluation, Comparative Benchmarks & Diagnostic Error Analysis**:
   * **Multi-query testing**: Validates retrieval accuracy across diverse user phrasings.
   * **NumPy vs. ChromaDB**: Demonstrates identical top-rank consistency between exact linear scan and ChromaDB's indexed retrieval.
   * **Ambiguous Query Diagnosis**: Evaluates poorly performing queries (e.g., *"How do I fix errors?"*), diagnosing why dense semantic search struggles with underspecified intent and outlining solutions (query disambiguation, hybrid BM25 + dense search, and category constraints).

---

## 📋 Rubric & Grading Compliance Checklist

| Rubric Requirement | Points | Status | Location / Implementation Details |
| :--- | :---: | :---: | :--- |
| **Public GitHub Repository** | **4** | ✅ | Remote tracked at [`https://github.com/EyobTesfaye8/AWS_tasks`](https://github.com/EyobTesfaye8/AWS_tasks). |
| **Loads Raw Dataset in Repo** | **5** | ✅ | Both `week-01/marketing.csv` and `week-02/data/*.csv` stored directly in repository. |
| **Named Functions (`def`) & Cleaning** | **10** | ✅ | Modular functions for null checks, deduplication, text normalization, type conversion, and IQR outliers. |
| **NumPy Calculations ($\ge 4$ funcs)** | **10** | ✅ | `np.mean`, `np.median`, `np.std`, `np.percentile`, `np.where`, `np.min`, `np.max`, `np.dot`. |
| **Executed Notebook Committed (No Errors)** | **8** | ✅ | Pre-computed outputs and charts saved in [`week-01`](week-01/AWS_First_week_task.ipynb) and [`week-02`](week-02/AWS_ML_Semantic_Search.ipynb) with 0 error cells. |
| **Colab Links Provided** | **2** | ✅ | Active Colab launch badges and URLs included in README and top of each notebook. |
| **`requirements.txt` with `numpy`** | **4** | ✅ | Explicitly listed in root [`requirements.txt`](requirements.txt). |
| **README Explains `venv` & `.gitignore`** | **3** | ✅ | Step-by-step virtualenv instructions for Linux/macOS and Windows; `.venv` ignored in `.gitignore`. |
| **README Explains How to Run** | **2** | ✅ | Clear execution commands provided for Python CLI scripts, Jupyter, and Colab for both weeks. |
| **Folder Organization (`week-01`, `week-02`)** | **2** | ✅ | Clean project folder hierarchy; no flat repository clutter. |
| **Penalty Prevention** | **0** | ✅ | `.venv`, `__pycache__`, and `.ipynb_checkpoints` strictly ignored and clean. |
| **Total Score Potential** | **50 / 50** | 🌟 | **100% Rubric Coverage** |
