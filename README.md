# Autonomous Data Analyst

**Multi-Agent AI Data Analysis System**

An autonomous multi-agent platform that takes a structured tabular dataset (CSV/XLSX) and — without any manual coding — produces: validated data understanding, safe cleaning, exploratory data analysis with visualizations, automatic ML model training (when applicable), verified business insights, an interactive dashboard, a PDF report, a PowerPoint presentation, and an optional natural-language chat interface over the results.

---

## Architecture

The system operates as a pipeline of specialized agents, each with a single responsibility, communicating through a shared **Dataset Intelligence Object (DIO)**:

| Agent | Role |
|-------|------|
| **Agent 1 — Intelligence** | Schema profiling, date resolution, semantic labeling, PII detection, domain classification, quality scoring |
| **Agent 2 — Cleaning** | Deterministic, reversible data cleaning with full transparency |
| **Agent 3 — EDA/Viz** | Rule-based exploratory data analysis and automatic chart generation |
| **Agent 4 — ML** | Automatic target detection, classification/regression training, feature importance |
| **Agent 5 — Insight** | LLM-powered business insight generation with hallucination verification |
| **Agent 6 — Dashboard/Report** | Interactive Streamlit dashboard, PDF report, PowerPoint presentation |
| **Agent 7 — Chat** | Optional natural-language Q&A over verified results (built last) |

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| **Default AI** | Ollama + Llama 3.1 8B (free, local, no API key) |
| **Application** | Streamlit |
| **Data Processing** | pandas, NumPy |
| **Machine Learning** | scikit-learn, XGBoost |
| **Visualization** | Plotly + Kaleido |
| **Reports** | ReportLab (PDF), python-pptx (PPTX) |
| **Persistence** | Filesystem + SQLite |
| **Testing** | pytest |

---

## V1 Analytical Scope

**Supported data:** Structured / tabular data (CSV, XLSX)

**Primary capabilities:**
- Data understanding and schema detection
- Data-type inference and semantic column classification
- Date/time detection with deterministic resolution
- PII detection and protection
- Domain classification
- Data-quality assessment
- Safe, reversible cleaning
- Exploratory Data Analysis with automatic visualization
- Automatic ML problem detection and target selection
- Classification and regression model training
- Model evaluation and feature importance
- Model verification
- Business insights with numerical grounding
- Interactive dashboard
- PDF report and PowerPoint presentation
- Natural-language querying over verified results

---

## Accuracy & Uncertainty

> **The system does not guarantee 100% accuracy.**
>
> It uses deterministic validation, confidence scoring, verification, and
> `UNKNOWN` / `UNCERTAIN` states where evidence is insufficient.
> All important numerical results originate from deterministic computation.
> LLMs may explain computed results, but must not invent them.
> The system is designed to be benchmarkable so actual accuracy can be measured.

---

## Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai/) with `llama3.1:8b` model
- Git
- Docker (optional, for containerized deployment)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd autonomous-data-analyst

# Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Ollama Setup

```bash
# Install Ollama from https://ollama.ai/
ollama pull llama3.1:8b
```

### Run

```bash
streamlit run app.py
```

### Docker (alternative)

```bash
docker compose up --build
```

Then open `http://localhost:8501` in your browser.

---

## Security

- PII is detected before any LLM interaction and **never** sent to an LLM
- No arbitrary LLM-generated code or SQL is executed
- All file uploads are validated before processing
- API keys are loaded from environment variables, never committed
- Cleaning operations are reversible; removed rows are preserved

---

## Limitations

- V1 supports CSV and XLSX only (JSON, Parquet, SQL planned for V2)
- ML is limited to classification and regression (no forecasting, clustering, or RL)
- No SHAP/LIME — uses built-in feature importances
- No cross-session memory or knowledge graphs
- Maximum upload size: 200 MB
- LLM token budget: 20,000 tokens per dataset run

---

## License

See LICENSE file for details.
