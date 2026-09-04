"""
agents/report/pdf_generator.py
==============================
Deterministic, executive PDF report generator using ReportLab.
Transforms verified Dataset Intelligence Object (DIO) data into a
clean, structured analytical report deliverable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from agents.report.data_extractor import extract_report_payload


class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically compute and draw total page numbers ('Page X of Y')
    along with running header and footer on each page.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._saved_page_states: list[dict[str, Any]] = []

    def showPage(self) -> None:
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self) -> None:
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count: int) -> None:
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#718096"))

        # Running Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(36, 756, "Autonomous Data Analyst — Analytical Intelligence Report")
            self.setStrokeColor(colors.HexColor("#E2E8F0"))
            self.setLineWidth(0.5)
            self.line(36, 750, 576, 750)

        # Running Footer (all pages)
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(36, 32, 576, 32)

        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(576, 20, page_str)
        self.drawString(36, 20, "Confidential — Automated Intelligence Deliverable")
        self.restoreState()


def _build_pdf_styles() -> dict[str, ParagraphStyle]:
    """Create a unified palette and typography style hierarchy."""
    base_styles = getSampleStyleSheet()

    styles: dict[str, ParagraphStyle] = {}

    styles["DocTitle"] = ParagraphStyle(
        "DocTitle",
        parent=base_styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=4,
    )
    styles["DocSubtitle"] = ParagraphStyle(
        "DocSubtitle",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#4A5568"),
        spaceAfter=12,
    )
    styles["SectionHeader"] = ParagraphStyle(
        "SectionHeader",
        parent=base_styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#1A365D"),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True,
    )
    styles["SubSectionHeader"] = ParagraphStyle(
        "SubSectionHeader",
        parent=base_styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True,
    )
    styles["Body"] = ParagraphStyle(
        "Body",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#2D3748"),
    )
    styles["BodyBold"] = ParagraphStyle(
        "BodyBold",
        parent=styles["Body"],
        fontName="Helvetica-Bold",
    )
    styles["TableHeader"] = ParagraphStyle(
        "TableHeader",
        parent=styles["BodyBold"],
        fontSize=8,
        leading=10,
        textColor=colors.white,
    )
    styles["TableCell"] = ParagraphStyle(
        "TableCell",
        parent=styles["Body"],
        fontSize=7.5,
        leading=10,
    )
    styles["TableCellBold"] = ParagraphStyle(
        "TableCellBold",
        parent=styles["BodyBold"],
        fontSize=7.5,
        leading=10,
    )
    styles["TableCellBadge"] = ParagraphStyle(
        "TableCellBadge",
        parent=styles["TableCell"],
        textColor=colors.HexColor("#C53030"),
        fontName="Helvetica-Bold",
    )
    styles["CalloutTitle"] = ParagraphStyle(
        "CalloutTitle",
        parent=styles["BodyBold"],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#1A365D"),
    )
    styles["CalloutText"] = ParagraphStyle(
        "CalloutText",
        parent=styles["Body"],
        fontSize=8,
        leading=11,
    )
    styles["KpiNumber"] = ParagraphStyle(
        "KpiNumber",
        parent=base_styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=18,
        alignment=1,  # Center
        textColor=colors.HexColor("#1A365D"),
    )
    styles["KpiLabel"] = ParagraphStyle(
        "KpiLabel",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=9,
        alignment=1,  # Center
        textColor=colors.HexColor("#718096"),
    )

    return styles


def generate_pdf_report(dio: Any, output_path: str | Path) -> Path:
    """
    Generate an analytical PDF report from DIO data.
    
    Parameters:
        dio: Dataset Intelligence Object (dataclass or dict)
        output_path: Path where the generated PDF will be saved
        
    Returns:
        Path to the saved PDF file.
    """
    dest_path = Path(output_path).resolve()
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Extract and sanitize DIO data
    data = extract_report_payload(dio)
    styles = _build_pdf_styles()

    # Printable width: letter is 612 x 792 pt. Margins 36 pt left/right -> 540 pt width.
    doc = SimpleDocTemplate(
        str(dest_path),
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=46,
        bottomMargin=42,
        pageCompression=0,  # Uncompressed streams for instant verification & inspection
    )

    story: list[Any] = []

    # -------------------------------------------------------------------------
    # 1. TITLE & EXECUTIVE HEADER
    # -------------------------------------------------------------------------
    file_name = data["metadata"]["file_name"]
    dataset_id = data["metadata"]["dataset_id"]
    gen_time = data["metadata"]["generated_at"]

    story.append(Paragraph("AUTONOMOUS DATA ANALYST", styles["DocSubtitle"]))
    story.append(Paragraph(f"Executive Intelligence Report: {file_name}", styles["DocTitle"]))
    meta_line = (
        f"<b>Dataset ID:</b> {dataset_id} &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"<b>Generated:</b> {gen_time} &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"<b>Engine:</b> Autonomous Data Analyst v1.0"
    )
    story.append(Paragraph(meta_line, styles["DocSubtitle"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1A365D"), spaceAfter=12))

    # -------------------------------------------------------------------------
    # 2. EXECUTIVE SUMMARY & KPI CARDS
    # -------------------------------------------------------------------------
    story.append(Paragraph("1. Executive Summary", styles["SectionHeader"]))

    # 5 KPI Callout Cards: Rows | Columns | Quality Score | Domain | ML Status
    q_score = data["quality"]["score"]
    domain_name = data["domain"]["name"]
    ml_status_text = data["ml"]["status"].upper()

    kpi_cells = [
        [
            Paragraph(f"{data['ingestion']['n_rows']:,}", styles["KpiNumber"]),
            Paragraph(f"{data['ingestion']['n_columns']}", styles["KpiNumber"]),
            Paragraph(f"{q_score}/100", styles["KpiNumber"]),
            Paragraph(domain_name, styles["KpiNumber"]),
            Paragraph(ml_status_text, styles["KpiNumber"]),
        ],
        [
            Paragraph("Total Ingested Rows", styles["KpiLabel"]),
            Paragraph("Total Features", styles["KpiLabel"]),
            Paragraph("Data Quality Score", styles["KpiLabel"]),
            Paragraph(f"Domain ({int(data['domain']['confidence']*100)}% conf)", styles["KpiLabel"]),
            Paragraph("ML Pipeline Status", styles["KpiLabel"]),
        ],
    ]
    kpi_table = Table(kpi_cells, colWidths=[108] * 5)
    kpi_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F7FAFC")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 8))

    exec_summary_text = (
        f"Automated intelligence ingestion and profiling completed for <b>{file_name}</b>. "
        f"The dataset comprises <b>{data['ingestion']['n_rows']} rows</b> across <b>{data['ingestion']['n_columns']} features</b>, "
        f"classified under the <b>{domain_name}</b> domain. Overall dataset quality is rated at <b>{q_score}/100</b>. "
    )
    if data["pii_columns"]:
        exec_summary_text += (
            f"Strict security protocols identified and isolated <b>{len(data['pii_columns'])} PII/identifier columns</b> "
            f"({', '.join(data['pii_columns'])}) to prevent privacy leakage. "
        )
    if data["insights"]:
        exec_summary_text += f"A total of <b>{len(data['insights'])} verified statistical and ML insights</b> were synthesized. "
    story.append(Paragraph(exec_summary_text, styles["Body"]))
    story.append(Spacer(1, 12))

    # -------------------------------------------------------------------------
    # 3. DATASET ARCHITECTURE & COLUMN CATALOG
    # -------------------------------------------------------------------------
    story.append(Paragraph("2. Dataset Architecture & Feature Catalog", styles["SectionHeader"]))
    catalog_intro = (
        "Complete schema inspection. To preserve privacy and security, raw cell values are strictly isolated; "
        "only verified structural metadata and logical inferences are reported below."
    )
    story.append(Paragraph(catalog_intro, styles["Body"]))
    story.append(Spacer(1, 6))

    catalog_headers = [
        Paragraph("Feature Name", styles["TableHeader"]),
        Paragraph("Logical Type", styles["TableHeader"]),
        Paragraph("Null Count", styles["TableHeader"]),
        Paragraph("Null %", styles["TableHeader"]),
        Paragraph("Cardinality", styles["TableHeader"]),
        Paragraph("Privacy & Semantic Classification", styles["TableHeader"]),
    ]
    catalog_rows = [catalog_headers]

    for col in data["columns"]:
        b_style = styles["TableCellBadge"] if (col["is_pii"] or col["is_identifier"]) else styles["TableCell"]
        catalog_rows.append([
            Paragraph(col["name"], styles["TableCellBold"]),
            Paragraph(col["logical_type"], styles["TableCell"]),
            Paragraph(str(col["null_count"]), styles["TableCell"]),
            Paragraph(f"{col['null_pct']}%", styles["TableCell"]),
            Paragraph(str(col["unique_count"]), styles["TableCell"]),
            Paragraph(f"{col['semantic_label']} — {col['status_badge']}", b_style),
        ])

    # Table width: 540 pt -> [120, 70, 55, 45, 60, 190]
    catalog_table = Table(catalog_rows, colWidths=[120, 70, 55, 45, 60, 190], repeatRows=1)
    catalog_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1A365D")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
    ]))
    story.append(catalog_table)
    story.append(Spacer(1, 12))

    # -------------------------------------------------------------------------
    # 4. DATA QUALITY & CLEANING OPERATIONS
    # -------------------------------------------------------------------------
    story.append(Paragraph("3. Data Quality Assessment & Cleaning Pipeline", styles["SectionHeader"]))
    
    dq_text = f"<b>Data Quality Score:</b> {q_score} / 100. "
    if data["quality"]["issues"]:
        dq_text += f"Detected issues: {'; '.join(data['quality']['issues'])}. "
    else:
        dq_text += "No critical data quality defects detected. "
    story.append(Paragraph(dq_text, styles["Body"]))
    story.append(Spacer(1, 6))

    cleaning_ops = data["cleaning"]["operations"]
    if cleaning_ops:
        story.append(Paragraph("<b>Executed Cleaning Transformations:</b>", styles["SubSectionHeader"]))
        for op in cleaning_ops:
            story.append(Paragraph(f"• {op}", styles["Body"]))
    else:
        story.append(Paragraph("• No destructive mutations required; dataset maintained structural cleanliness.", styles["Body"]))
    story.append(Spacer(1, 12))

    # -------------------------------------------------------------------------
    # 5. EXPLORATORY DATA ANALYSIS (EDA)
    # -------------------------------------------------------------------------
    story.append(Paragraph("4. Exploratory Data Analysis (EDA)", styles["SectionHeader"]))

    # Summary Statistics Table
    if data["eda"]["summary_stats"]:
        story.append(Paragraph("<b>Key Numeric Distribution Statistics:</b>", styles["SubSectionHeader"]))
        stats_headers = [
            Paragraph("Feature", styles["TableHeader"]),
            Paragraph("Mean", styles["TableHeader"]),
            Paragraph("Std Dev", styles["TableHeader"]),
            Paragraph("Min", styles["TableHeader"]),
            Paragraph("Median", styles["TableHeader"]),
            Paragraph("Max", styles["TableHeader"]),
        ]
        stats_rows = [stats_headers]
        for s in data["eda"]["summary_stats"]:
            stats_rows.append([
                Paragraph(s["column"], styles["TableCellBold"]),
                Paragraph(s["mean"], styles["TableCell"]),
                Paragraph(s["std"], styles["TableCell"]),
                Paragraph(s["min"], styles["TableCell"]),
                Paragraph(s["median"], styles["TableCell"]),
                Paragraph(s["max"], styles["TableCell"]),
            ])
        # Width: [140, 80, 80, 80, 80, 80] = 540 pt
        stats_table = Table(stats_rows, colWidths=[140, 80, 80, 80, 80, 80], repeatRows=1)
        stats_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ]))
        story.append(stats_table)
        story.append(Spacer(1, 8))

    # Top Correlations
    if data["eda"]["top_correlations"]:
        story.append(Paragraph("<b>Significant Feature Correlations:</b>", styles["SubSectionHeader"]))
        corr_headers = [
            Paragraph("Feature A", styles["TableHeader"]),
            Paragraph("Feature B", styles["TableHeader"]),
            Paragraph("Pearson Correlation (r)", styles["TableHeader"]),
        ]
        corr_rows = [corr_headers]
        for c in data["eda"]["top_correlations"][:6]:
            corr_rows.append([
                Paragraph(c["feature_a"], styles["TableCellBold"]),
                Paragraph(c["feature_b"], styles["TableCell"]),
                Paragraph(f"{c['correlation']:+.3f}", styles["TableCell"]),
            ])
        # Width: [200, 200, 140] = 540 pt
        corr_table = Table(corr_rows, colWidths=[200, 200, 140], repeatRows=1)
        corr_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ]))
        story.append(corr_table)
        story.append(Spacer(1, 8))

    # Visualizations (Embedded Charts)
    charts = data["eda"]["chart_paths"]
    if charts:
        story.append(Paragraph("<b>Key Visualizations:</b>", styles["SubSectionHeader"]))
        for idx, cp in enumerate(charts[:4], start=1):
            try:
                from PIL import Image as PILImage
                with PILImage.open(cp) as im:
                    im.verify()
                # 480 x 260 pt image display with aspect ratio preserved
                img = Image(cp, width=480, height=260)
                img.hAlign = "CENTER"
                caption = Paragraph(f"<i>Figure {idx}: {Path(cp).stem.replace('_', ' ').title()}</i>", styles["KpiLabel"])
                story.append(KeepTogether([img, Spacer(1, 3), caption, Spacer(1, 10)]))
            except Exception:
                story.append(Paragraph(f"<i>[Visualization {Path(cp).name} could not be rendered inline]</i>", styles["Body"]))
    else:
        story.append(Paragraph("<i>No graphical charts were generated for this run.</i>", styles["Body"]))
    story.append(Spacer(1, 12))

    # -------------------------------------------------------------------------
    # 6. MACHINE LEARNING RESULTS
    # -------------------------------------------------------------------------
    story.append(Paragraph("5. Machine Learning Analysis", styles["SectionHeader"]))
    ml_stat = data["ml"]["status"]

    if ml_stat == "trained":
        ml_overview = (
            f"Supervised machine learning completed successfully. "
            f"<b>Task:</b> {data['ml']['task_type']} &nbsp;|&nbsp; "
            f"<b>Target Variable:</b> '{data['ml']['target_column']}' &nbsp;|&nbsp; "
            f"<b>Selected Optimal Model:</b> {data['ml']['best_model']}"
        )
        story.append(Paragraph(ml_overview, styles["Body"]))
        story.append(Spacer(1, 6))

        # Metrics Table
        if data["ml"]["metrics"]:
            story.append(Paragraph("<b>Out-of-Sample Evaluation Metrics:</b>", styles["SubSectionHeader"]))
            m_headers = [Paragraph("Metric", styles["TableHeader"]), Paragraph("Evaluated Score", styles["TableHeader"])]
            m_rows = [m_headers]
            for m_name, m_score in data["ml"]["metrics"].items():
                m_rows.append([Paragraph(m_name, styles["TableCellBold"]), Paragraph(str(m_score), styles["TableCell"])])
            m_table = Table(m_rows, colWidths=[270, 270])
            m_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ]))
            story.append(m_table)
            story.append(Spacer(1, 8))

        # Feature Importance Table
        if data["ml"]["feature_importances"]:
            story.append(Paragraph("<b>Top Predictive Feature Drivers:</b>", styles["SubSectionHeader"]))
            fi_headers = [Paragraph("Predictive Feature", styles["TableHeader"]), Paragraph("Relative Importance", styles["TableHeader"])]
            fi_rows = [fi_headers]
            for f_name, f_imp in data["ml"]["feature_importances"]:
                fi_rows.append([Paragraph(f_name, styles["TableCellBold"]), Paragraph(f"{f_imp:.4f}", styles["TableCell"])])
            fi_table = Table(fi_rows, colWidths=[270, 270])
            fi_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ]))
            story.append(fi_table)

    elif ml_stat == "insufficient_data":
        notice = (
            "<b>Machine Learning Notice:</b> Predictive modeling was safely bypassed due to insufficient sample size. "
            f"The dataset does not satisfy the required minimum row threshold for training. Reason: <i>{data['ml']['reason']}</i>"
        )
        story.append(Paragraph(notice, styles["Body"]))
    elif ml_stat in ("unsupported", "no_target"):
        notice = (
            "<b>Machine Learning Notice:</b> Supervised modeling was not initiated because no valid predictive "
            f"target variable was identified. Reason: <i>{data['ml']['reason'] or 'No target variable detected.'}</i>"
        )
        story.append(Paragraph(notice, styles["Body"]))
    elif ml_stat == "leakage_detected":
        notice = (
            "<b>Machine Learning Warning:</b> Model training was aborted by the autonomous data leakage guard. "
            f"High target correlation or post-outcome leakage detected: <i>{data['ml']['reason']}</i>"
        )
        story.append(Paragraph(notice, styles["Body"]))
    else:
        story.append(Paragraph("<i>Machine learning modeling was not configured or executed for this analysis.</i>", styles["Body"]))
    story.append(Spacer(1, 12))

    # -------------------------------------------------------------------------
    # 7. KEY BUSINESS INSIGHTS
    # -------------------------------------------------------------------------
    story.append(Paragraph("6. Synthesized Business Insights", styles["SectionHeader"]))
    if data["insights"]:
        story.append(Paragraph(
            "Every insight below has passed rigorous statistical verification and numerical grounding. "
            "Hallucinated claims are strictly filtered out by the Insight Guard.",
            styles["Body"]
        ))
        story.append(Spacer(1, 6))

        for ins in data["insights"]:
            card_content = [
                [Paragraph(f"<b>[{ins['category'].upper()}]</b> Insight {ins['id']} (Confidence: {int(ins['confidence']*100)}%)", styles["CalloutTitle"])],
                [Paragraph(ins["text"], styles["CalloutText"])],
                [Paragraph(f"<b>Evidence:</b> {ins['evidence']}", styles["KpiLabel"])],
            ]
            if ins["recommendation"]:
                card_content.append([Paragraph(f"<b>Action:</b> {ins['recommendation']}", styles["CalloutText"])])

            card_table = Table(card_content, colWidths=[540])
            card_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F7FAFC")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(KeepTogether([card_table, Spacer(1, 6)]))
    else:
        story.append(Paragraph("<i>Zero verified insights could be grounded from this dataset with high confidence.</i>", styles["Body"]))
    story.append(Spacer(1, 12))

    # -------------------------------------------------------------------------
    # 8. ACTIONABLE RECOMMENDATIONS
    # -------------------------------------------------------------------------
    story.append(Paragraph("7. Actionable Recommendations", styles["SectionHeader"]))
    if data["recommendations"]:
        for idx, rec in enumerate(data["recommendations"], start=1):
            story.append(Paragraph(f"<b>{idx}.</b> {rec}", styles["Body"]))
            story.append(Spacer(1, 3))
    else:
        story.append(Paragraph("<i>No immediate strategic actions recommended based on current data volume.</i>", styles["Body"]))
    story.append(Spacer(1, 12))

    # -------------------------------------------------------------------------
    # 9. LIMITATIONS & WARNINGS
    # -------------------------------------------------------------------------
    story.append(Paragraph("8. Analytical Limitations & Guardrail Warnings", styles["SectionHeader"]))
    if data["limitations"]:
        for lim in data["limitations"]:
            story.append(Paragraph(f"• <b>Notice:</b> {lim}", styles["Body"]))
            story.append(Spacer(1, 2))
    else:
        story.append(Paragraph("• No significant technical or data quality limitations noted.", styles["Body"]))
    story.append(Spacer(1, 12))

    # -------------------------------------------------------------------------
    # 10. TECHNICAL METADATA & PROVENANCE
    # -------------------------------------------------------------------------
    story.append(Paragraph("9. Technical Metadata & Provenance", styles["SectionHeader"]))
    meta_rows = [
        [Paragraph("Run / Dataset ID", styles["TableCellBold"]), Paragraph(data["metadata"]["dataset_id"], styles["TableCell"])],
        [Paragraph("Source File Name", styles["TableCellBold"]), Paragraph(data["metadata"]["file_name"], styles["TableCell"])],
        [Paragraph("Dataset SHA-256 Hash", styles["TableCellBold"]), Paragraph(data["metadata"]["dataset_hash"] or "Not Calculated", styles["TableCell"])],
        [Paragraph("DIO Schema Version", styles["TableCellBold"]), Paragraph(data["metadata"]["schema_version"], styles["TableCell"])],
        [Paragraph("Generation Timestamp", styles["TableCellBold"]), Paragraph(data["metadata"]["generated_at"], styles["TableCell"])],
        [Paragraph("Pipeline Execution Status", styles["TableCellBold"]), Paragraph("Report Generation Completed Successfully", styles["TableCell"])],
    ]
    meta_table = Table(meta_rows, colWidths=[160, 380])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EDF2F7")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
    ]))
    story.append(meta_table)

    # Build PDF with NumberedCanvas
    doc.build(story, canvasmaker=NumberedCanvas)
    return dest_path
