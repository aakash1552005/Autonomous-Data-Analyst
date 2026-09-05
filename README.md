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

### Run Application

Launch the executive Streamlit user interface:

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## Pipeline Workflow (Phase 9)

The system orchestrates a 7-stage analytical pipeline coordinated by `orchestrator.py`:

```text
Upload (.csv / .xlsx)
  │
  ▼
[Stage 1: Validation & Ingestion] (FATAL)
  ├── Security file inspection, size validation, CSV/Excel parsing
  └── Dataset Intelligence Object (DIO) initialization & run directory setup
  │
  ▼
[Stage 2: Dataset Intelligence] (RECOVERABLE)
  ├── Structural schema profiling & type inference
  ├── Date normalization / preservation of ambiguous formats
  ├── PII detection & semantic labeling
  └── Domain classification & data quality assessment
  │
  ▼
[Stage 3: Data Cleaning] (FATAL)
  ├── Deterministic duplicate row isolation (sidecar preservation)
  ├── Inferred type coercion & date normalization
  ├── Numeric & categorical missing value imputation
  └── Non-destructive outlier flagging
  │
  ▼
[Stage 4: Exploratory Data Analysis & Viz] (RECOVERABLE)
  ├── Descriptive statistical profiles & skewness calculation
  ├── Correlation matrix analysis
  └── Publication-grade visual chart artifacts (PNG)
  │
  ▼
[Stage 5: Machine Learning Modeling] (RECOVERABLE)
  ├── Automatic target variable recommendation & problem typing
  ├── Train/test validation with stratified/k-fold CV
  ├── Model candidate evaluation (Linear/Logistic, Tree, Random Forest, XGBoost)
  └── Feature importance extraction & verification checks (or safe skipping)
  │
  ▼
[Stage 6: Executive Insights] (RECOVERABLE)
  ├── Multi-source evidence compilation
  ├── LLM narrative synthesis with deterministic fallback
  └── Hallucination guard: 100% numerical grounding verification
  │
  ▼
[Stage 7: Deliverable Generation] (RECOVERABLE)
  ├── Executive PDF Report (ReportLab) with executive summary, metrics, charts, insights
  └── Executive Presentation Deck (.pptx) with structured slide hierarchy
```

### Failure Classification & Status Semantics

The pipeline strictly categorizes stage outcomes to ensure robustness:

- **FATAL Stages** (`Validation`, `Cleaning`): If an unrecoverable error occurs (e.g. unsupported file type, corrupt file, zero rows, or cleaning failure), the pipeline immediately halts, records the failure, and returns overall status `failed`.
- **RECOVERABLE Stages** (`Intelligence`, `EDA`, `ML`, `Insight`, `Report`): If an unexpected exception occurs, the failure is explicitly recorded in DIO `errors` and stage telemetry without swallowing. Safe downstream stages continue, and the overall pipeline status is marked `partial`.
- **Completed**: Only runs where all stages succeed without fatal or recoverable failures receive status `completed`.

---

## Configuration & Customization

The pipeline is configured via `config.yaml` with environment variable overrides:

```yaml
max_upload_size_mb: 200

llm:
  provider: ollama             # ollama | openai
  model: llama3.1:8b
  host: http://127.0.0.1:11434
  timeout_seconds: 60

security:
  max_llm_tokens_per_run: 20000

eda:
  max_charts: 6
  correlation_threshold: 0.3

ml:
  max_training_time_seconds: 60
  cv_folds: 5

insights:
  min_insights: 3
  max_insights: 6
  grounding_tolerance: 0.05
  temperature: 0.2
  max_tokens: 800

pipeline:
  runs_dir: runs
  save_dio_json: true
  log_level: INFO
```

---

## Generated Artifacts & Deliverables

Every analysis run produces an isolated run directory (`runs/{timestamp}_{dataset_name}/`) containing:

| Artifact | Format | Description |
|---|---|---|
| `cleaned_data.csv` | CSV | Reversibly cleaned dataset with types coerced and missing values imputed |
| `removed_rows.csv` | CSV | Duplicate rows safely extracted into a sidecar (if duplicates existed) |
| `charts/*.png` | PNG | Publication-grade charts generated during EDA |
| `best_model.joblib` | Binary | Serialized scikit-learn/XGBoost model pipeline (when ML target exists) |
| `report.pdf` | PDF | Executive multi-page analytical report with charts and grounded insights |
| `presentation.pptx` | PPTX | Executive presentation slide deck |
| `dio.json` | JSON | Complete Dataset Intelligence Object state capturing all analytical findings |
| `pipeline.log` | Text | Execution log recording stage durations, events, and warnings |

---

## Security & PII Protection

- PII is detected before any LLM interaction and **never** sent to an LLM.
- Streamlit data previews redact all identified PII columns.
- LLM prompt payloads are strictly filtered and masked using `mask_value_str`.
- All file uploads are validated for security, MIME types, and path traversal guards.
- API keys are loaded strictly from environment variables and redacted from logs.
- Cleaning operations are non-destructive and fully reversible.

---

## License

See LICENSE file for details.
