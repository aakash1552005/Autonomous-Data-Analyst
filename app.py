"""
app.py
======
Phase 9: Streamlit User Interface for Autonomous Data Analyst.

Provides an executive-grade dashboard for:
  - Dataset upload (.csv, .xlsx) with security validation
  - Real-time pipeline progress tracking
  - In-depth analytical results (Intelligence, Cleaning, EDA, ML, Insights)
  - Interactive chart inspection
  - Direct download of executive PDF reports, PPTX decks, cleaned CSVs, model artifacts, and DIO state
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from core.config import AppConfig, load_config
from core.dio import DIO
from orchestrator import Orchestrator, OrchestratorResult, PIPELINE_STAGES
from utils.mask_for_llm import mask_value_str


def init_page_config() -> None:
    """Configure Streamlit page headers, layout, and styling."""
    st.set_page_config(
        page_title="Autonomous Data Analyst",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    # Custom CSS for executive styling
    st.markdown(
        """
        <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1E293B;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.05rem;
            color: #64748B;
            margin-bottom: 1.5rem;
        }
        .metric-card {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
        }
        .insight-card {
            background-color: #FFFFFF;
            border-left: 4px solid #3B82F6;
            border-top: 1px solid #E2E8F0;
            border-right: 1px solid #E2E8F0;
            border-bottom: 1px solid #E2E8F0;
            border-radius: 6px;
            padding: 1.2rem;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        .badge {
            display: inline-block;
            padding: 0.25rem 0.6rem;
            font-size: 0.8rem;
            font-weight: 600;
            border-radius: 4px;
            background-color: #EFF6FF;
            color: #1D4ED8;
            margin-bottom: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown('<div class="main-header">Autonomous Data Analyst</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">End-to-End Tabular Intelligence, Deterministic Cleaning, ML Modeling & Executive Reporting</div>',
        unsafe_allow_html=True,
    )


def safe_preview_dataframe(df: pd.DataFrame, dio: DIO, n_rows: int = 5) -> pd.DataFrame:
    """
    Produce a safe preview of the DataFrame with any PII columns masked.
    """
    preview = df.head(n_rows).copy()
    pii_cols = set()
    for col_info in dio.get("columns", []):
        if col_info.get("is_pii"):
            pii_cols.add(col_info.get("name"))

    for col in preview.columns:
        if col in pii_cols:
            preview[col] = preview[col].apply(lambda v: mask_value_str(v, is_pii=True))
    return preview


def run_pipeline(
    uploaded_file: Any,
    preferred_target: str | None,
    config: AppConfig,
) -> None:
    """Handle uploaded file, execute orchestrator pipeline, and update session state."""
    # Staging upload directory
    staging_dir = Path("runs") / "staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    temp_path = staging_dir / uploaded_file.name

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Progress UI containers
    progress_bar = st.progress(0.0)
    status_text = st.empty()

    def update_progress(stage: str, pct: float, message: str) -> None:
        progress_bar.progress(min(1.0, max(0.0, pct)))
        status_text.info(f"[{stage.upper()}] {message}")

    orchestrator = Orchestrator(config=config, progress_callback=update_progress)

    with st.spinner("Autonomous analysis pipeline in progress..."):
        result: OrchestratorResult = orchestrator.run(
            file_path=temp_path,
            preferred_target=preferred_target if preferred_target else None,
            progress_callback=update_progress,
        )

    # Persist in session state to prevent rerun loss
    st.session_state["pipeline_result"] = result
    st.session_state["file_name"] = uploaded_file.name
    progress_bar.progress(1.0)
    if result.status == "completed":
        status_text.success("Analysis complete! Review the analytical sections below.")
    elif result.status == "partial":
        status_text.warning("Analysis finished with partial status. Some optional stages encountered warnings.")
    else:
        status_text.error("Analysis failed. Review errors in the report below.")


def render_dashboard(result: OrchestratorResult, file_name: str) -> None:
    """Render comprehensive analytical dashboard from OrchestratorResult."""
    dio = result.dio
    if dio is None:
        st.error("No DIO output available. Execution failed during initial validation.")
        if result.errors:
            st.write(result.errors)
        return

    # Top-level Metric Banner
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Total Rows", f"{dio.get('ingestion', {}).get('n_rows', 0):,}")
    with col2:
        st.metric("Columns", f"{dio.get('ingestion', {}).get('n_columns', 0):,}")
    with col3:
        domain = dio.get("domain_guess", {}).get("domain", "Unknown").title()
        conf = dio.get("domain_guess", {}).get("confidence", 0.0)
        st.metric("Domain", domain, f"{int(conf * 100)}% conf")
    with col4:
        score = dio.get("quality", {}).get("score", 0)
        st.metric("Quality Score", f"{score} / 100")
    with col5:
        st.metric("Status", result.status.upper())
    with col6:
        runtime = result.stage_timings.get("total_pipeline", 0.0)
        st.metric("Runtime", f"{runtime:.1f}s")

    st.markdown("---")

    # Main Analytical Tabs
    tabs = st.tabs([
        "Dataset Overview",
        "Security & Quality",
        "Data Cleaning",
        "EDA & Visualizations",
        "Machine Learning",
        "Executive Insights",
        "Deliverables & Exports",
    ])

    # -------------------------------------------------------------
    # TAB 1: Overview & Profiling
    # -------------------------------------------------------------
    with tabs[0]:
        st.subheader("Dataset Ingestion & Schema Profiling")
        ingestion = dio.get("ingestion", {})
        c1, c2, c3 = st.columns(3)
        c1.write(f"**File Name:** `{file_name}`")
        c1.write(f"**File Type:** `{ingestion.get('file_type', 'N/A')}`")
        c2.write(f"**Encoding:** `{ingestion.get('encoding', 'utf-8')}`")
        c2.write(f"**Dataset Hash:** `{dio.get('dataset_hash', '')[:16]}...`")
        c3.write(f"**Run ID:** `{dio.get('dataset_id', '')}`")

        st.markdown("#### Column Inventory & Inferred Types")
        cols_data = []
        for col in dio.get("columns", []):
            cols_data.append({
                "Column": col.get("name"),
                "Raw Type": col.get("dtype"),
                "Semantic Type": col.get("semantic_type"),
                "Inferred Role": col.get("inferred_type"),
                "Null Count": col.get("null_count", 0),
                "Null %": f"{col.get('null_percentage', 0.0):.1f}%",
                "Unique Count": col.get("unique_count", 0),
                "Is PII": "⚠️ Yes" if col.get("is_pii") else "No",
            })
        if cols_data:
            st.dataframe(pd.DataFrame(cols_data), use_container_width=True)

        if result.df is not None:
            st.markdown("#### Safe Masked Data Preview (Top 5 Rows)")
            preview = safe_preview_dataframe(result.df, dio, n_rows=5)
            st.dataframe(preview, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 2: Security & Quality
    # -------------------------------------------------------------
    with tabs[1]:
        st.subheader("Security, PII Boundary & Quality Assessment")

        q_col1, q_col2 = st.columns([1, 2])
        with q_col1:
            q_score = dio.get("quality", {}).get("score", 0)
            st.metric("Overall Quality Score", f"{q_score} / 100")
            if q_score >= 80:
                st.success("High data quality — suitable for modeling.")
            elif q_score >= 50:
                st.warning("Moderate quality — cleaning and imputation applied.")
            else:
                st.error("Low quality dataset — substantial anomalies present.")

        with q_col2:
            st.markdown("**Identified Data Issues:**")
            issues = dio.get("quality", {}).get("issues", [])
            if issues:
                for issue in issues:
                    st.write(f"- {issue}")
            else:
                st.info("No major data quality defects flagged.")

        st.markdown("---")
        st.markdown("#### Sensitive Entities & PII Detection")
        pii_records = []
        for col in dio.get("columns", []):
            if col.get("is_pii"):
                pii_records.append({
                    "Column": col.get("name"),
                    "PII Type": col.get("pii_type", "Generic PII"),
                    "Confidence": col.get("pii_confidence", 1.0),
                    "Action Taken": "Redacted from LLM prompts and reports",
                })
        if pii_records:
            st.dataframe(pd.DataFrame(pii_records), use_container_width=True)
        else:
            st.success("No sensitive Personally Identifiable Information (PII) detected.")

    # -------------------------------------------------------------
    # TAB 3: Data Cleaning
    # -------------------------------------------------------------
    with tabs[2]:
        st.subheader("Deterministic Cleaning & Transformations")
        cleaning_logs = dio.get("cleaning_log", [])
        if cleaning_logs:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Executed Cleaning Actions:**")
                for item in cleaning_logs:
                    action = item.get("action", item.get("step", "Transformation"))
                    details = item.get("details", item.get("description", ""))
                    st.write(f"- **{action}:** {details}")
            with c2:
                artifacts = dio.get("artifacts", {})
                cleaned_csv = artifacts.get("cleaned_csv")
                removed_csv = artifacts.get("removed_rows_csv")
                if cleaned_csv and Path(cleaned_csv).exists():
                    st.download_button(
                        label="📥 Download Cleaned Dataset (CSV)",
                        data=Path(cleaned_csv).read_bytes(),
                        file_name=Path(cleaned_csv).name,
                        mime="text/csv",
                    )
                if removed_csv and Path(removed_csv).exists():
                    st.download_button(
                        label="📥 Download Removed Duplicate Rows (CSV)",
                        data=Path(removed_csv).read_bytes(),
                        file_name=Path(removed_csv).name,
                        mime="text/csv",
                    )
        else:
            st.info("No cleaning transformations were necessary or recorded.")

    # -------------------------------------------------------------
    # TAB 4: EDA & Visualizations
    # -------------------------------------------------------------
    with tabs[3]:
        st.subheader("Exploratory Data Analysis")
        eda = dio.get("eda", {})

        # Summary statistics
        if eda.get("summary_stats"):
            st.markdown("#### Numerical Summary Statistics")
            stats_df = pd.DataFrame(eda["summary_stats"])
            st.dataframe(stats_df, use_container_width=True)

        # Charts gallery
        chart_paths = dio.get("artifacts", {}).get("chart_paths", [])
        if not chart_paths and result.run_dir:
            chart_dir = result.run_dir / "charts"
            if chart_dir.exists():
                chart_paths = [str(p) for p in chart_dir.glob("*.png")]

        if chart_paths:
            st.markdown("#### Generated Visualizations")
            chart_cols = st.columns(2)
            for idx, cpath in enumerate(chart_paths):
                p = Path(cpath)
                if p.exists():
                    with chart_cols[idx % 2]:
                        st.image(str(p), caption=p.stem.replace("_", " ").title(), use_container_width=True)
        else:
            st.info("No visualization artifacts generated for this dataset.")

    # -------------------------------------------------------------
    # TAB 5: Machine Learning
    # -------------------------------------------------------------
    with tabs[4]:
        st.subheader("Machine Learning Modeling & Evaluation")
        ml_data = dio.get("ml", {})
        prob_type = ml_data.get("problem_type", "none")

        if prob_type == "none":
            st.info(
                "Machine learning modeling was skipped: dataset does not meet criteria for automated target modeling "
                "(no viable target column identified or insufficient variance)."
            )
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Task Type", prob_type.title())
            m2.metric("Target Variable", ml_data.get("target_column", "N/A"))
            m3.metric("Best Model", ml_data.get("best_model", "N/A"))

            # Models table
            models_tried = ml_data.get("models_tried", [])
            if models_tried:
                st.markdown("#### Candidate Models Comparison")
                st.dataframe(pd.DataFrame(models_tried), use_container_width=True)

            # Feature importance
            feat_imp = ml_data.get("feature_importance", {})
            if feat_imp:
                st.markdown("#### Feature Importance")
                imp_df = pd.DataFrame(
                    [{"Feature": k, "Importance": v} for k, v in feat_imp.items()]
                ).sort_values("Importance", ascending=False)
                st.bar_chart(imp_df.set_index("Feature"))

            # Download model artifact
            model_pkl = dio.get("artifacts", {}).get("model_pkl")
            if model_pkl and Path(model_pkl).exists():
                st.download_button(
                    label="📥 Download Trained Model (.joblib)",
                    data=Path(model_pkl).read_bytes(),
                    file_name=Path(model_pkl).name,
                    mime="application/octet-stream",
                )

    # -------------------------------------------------------------
    # TAB 6: Executive Insights
    # -------------------------------------------------------------
    with tabs[5]:
        st.subheader("Synthesized Executive Insights")
        insights = dio.get("insights", [])

        if not insights:
            st.info("No executive insights synthesized.")
        else:
            for idx, item in enumerate(insights):
                if isinstance(item, dict):
                    category = item.get("category", "General").title()
                    text = item.get("insight", item.get("text", ""))
                    evidence = item.get("evidence", [])
                    recom = item.get("recommendation", "")
                    conf = item.get("confidence", 0.9)

                    st.markdown(
                        f"""
                        <div class="insight-card">
                            <span class="badge">{category}</span>
                            <h4>Insight {idx + 1}</h4>
                            <p style="font-size: 1.05rem; color: #1E293B;">{text}</p>
                            <p style="color: #475569;"><strong>Confidence:</strong> {int(conf * 100)}%</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if evidence:
                        with st.expander(f"Supporting Evidence for Insight {idx + 1}"):
                            for ev in evidence:
                                st.write(f"- {ev}")
                    if recom:
                        st.info(f"💡 **Recommendation:** {recom}")
                else:
                    st.markdown(f"- {item}")

    # -------------------------------------------------------------
    # TAB 7: Deliverables & Exports
    # -------------------------------------------------------------
    with tabs[6]:
        st.subheader("Deliverables & Exportable Artifacts")
        artifacts = dio.get("artifacts", {})

        d1, d2 = st.columns(2)
        with d1:
            st.markdown("#### Executive Reports")
            pdf_path = artifacts.get("pdf_report")
            if pdf_path and Path(pdf_path).exists():
                st.download_button(
                    label="📄 Download Executive PDF Report",
                    data=Path(pdf_path).read_bytes(),
                    file_name=Path(pdf_path).name,
                    mime="application/pdf",
                    use_container_width=True,
                )
            else:
                st.write("PDF report not available.")

            pptx_path = artifacts.get("pptx_report")
            if pptx_path and Path(pptx_path).exists():
                st.download_button(
                    label="📊 Download Presentation Deck (.pptx)",
                    data=Path(pptx_path).read_bytes(),
                    file_name=Path(pptx_path).name,
                    mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    use_container_width=True,
                )
            else:
                st.write("PPTX presentation not available.")

        with d2:
            st.markdown("#### Data & State Artifacts")
            if result.run_dir:
                dio_json_path = result.run_dir / "dio.json"
                if dio_json_path.exists():
                    st.download_button(
                        label="💾 Download Dataset Intelligence Object (DIO JSON)",
                        data=dio_json_path.read_bytes(),
                        file_name="dio.json",
                        mime="application/json",
                        use_container_width=True,
                    )

                log_path = result.run_dir / "pipeline.log"
                if log_path.exists():
                    st.download_button(
                        label="📜 Download Execution Pipeline Log",
                        data=log_path.read_bytes(),
                        file_name="pipeline.log",
                        mime="text/plain",
                        use_container_width=True,
                    )


def main() -> None:
    init_page_config()
    render_header()
    config = load_config()

    # Sidebar Controls
    with st.sidebar:
        st.header("Pipeline Configuration")
        st.write(f"**Max Upload:** `{config.max_upload_size_mb} MB`")
        st.write(f"**LLM Provider:** `{config.llm.provider.upper()}`")
        st.write(f"**LLM Model:** `{config.llm.model}`")
        st.markdown("---")

        uploaded_file = st.file_uploader(
            "Upload Tabular Dataset",
            type=["csv", "xlsx"],
            help="Select a CSV or Excel dataset to begin end-to-end analysis.",
        )

        preferred_target = st.text_input(
            "Preferred ML Target Column (Optional)",
            help="Specify column name to target for machine learning. If left blank, optimal target is auto-detected.",
        )

        run_button = st.button("🚀 Analyze Dataset", type="primary", use_container_width=True)

        if st.session_state.get("pipeline_result") is not None:
            if st.button("🔄 Clear & Reset", use_container_width=True):
                st.session_state.pop("pipeline_result", None)
                st.session_state.pop("file_name", None)
                st.rerun()

    # Main Execution / Dashboard Routing
    if run_button:
        if uploaded_file is None:
            st.error("Please upload a CSV or XLSX file to begin analysis.")
        else:
            run_pipeline(uploaded_file, preferred_target, config)

    if st.session_state.get("pipeline_result") is not None:
        render_dashboard(
            st.session_state["pipeline_result"],
            st.session_state.get("file_name", "dataset"),
        )
    elif not run_button:
        st.info("👈 Upload a dataset in the sidebar and click **Analyze Dataset** to launch the pipeline.")


if __name__ == "__main__":
    main()
