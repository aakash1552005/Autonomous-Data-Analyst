# ERROR CODES REFERENCE
# Autonomous Data Analyst

This document defines the standardized error codes used throughout the platform.
Errors are prefixed by the producing subsystem or agent for rapid debugging and provenance tracking.

| Error Code | Subsystem / Agent | Description |
|------------|-------------------|-------------|
| `SYS_001`  | Core System       | Unexpected runtime or internal system error |
| `CFG_001`  | Configuration     | Configuration file missing, unparseable, or invalid |
| `DIO_001`  | Core DIO          | Dataset Intelligence Object validation, serialization, or deserialization error |
| `SEC_001`  | Security          | File validation failed (invalid extension, MIME type, or signature) |
| `SEC_002`  | Security          | File size exceeds maximum allowable limit |
| `SEC_003`  | Security          | Path traversal or insecure file access detected |
| `LLM_001`  | LLM Provider      | Connection failed, timeout, or provider returned an error |
| `LLM_002`  | LLM Governor      | Global per-run token budget exceeded; further calls blocked |
| `LLM_003`  | LLM Provider      | Missing API key for configured non-default provider |
| `INT_001`  | Intelligence      | Schema profiling or tabular structure parsing failure |
| `INT_002`  | Intelligence      | Date resolution ambiguity or parsing failure |
| `INT_003`  | Intelligence      | Semantic column labeling failure |
| `INT_004`  | Intelligence      | PII detection or masking failure |
| `CLN_001`  | Cleaning          | Type coercion or missing value imputation failure |
| `CLN_002`  | Cleaning          | Reversible record preservation failure |
| `EDA_001`  | EDA / Viz         | Summary statistics or correlation matrix computation error |
| `EDA_002`  | EDA / Viz         | Chart generation or image export failure |
| `ML_001`   | Machine Learning  | Target detection scoring error |
| `ML_002`   | Machine Learning  | Feature preprocessing or encoding failure |
| `ML_003`   | Machine Learning  | Model training or cross-validation failure |
| `ML_004`   | Machine Learning  | Model verification or post-training check failed |
| `INS_001`  | Insights          | Insight generation error |
| `INS_002`  | Insights          | Insight hallucination / numerical grounding verification rejected claim |
| `DSH_001`  | Dashboard         | KPI calculation or dashboard rendering error |
| `RPT_001`  | Report            | PDF report compilation failure |
| `RPT_002`  | Report            | PowerPoint presentation generation failure |
| `CHT_001`  | Chat              | Natural language query parsing or whitelisted execution error |
