# Agentic Code Intelligence - Phase 1 Prototype

## Overview
This repository contains the Phase 1 prototype for the **Agentic Code Intelligence** project. The solution is engineered for high-accuracy code retrieval over large, evolving codebases under strict minimal GPU resource constraints (100% CPU-optimized execution).

## Key Deliverables Completed

1. **Role 1: Query Understanding (`src/preprocessor.py`)**
   - **Rule-Based Intent Classifier**: Utilizes specialized regex patterns to classify user queries into three primary intents:
     - `definition-lookup`: Identifying function, class, interface, and struct definitions, signatures, and declarations.
     - `usage-lookup`: Identifying examples, function calls, imports, and invocation patterns.
     - `behavior-lookup`: Identifying processing workflows, runtime behaviors, logic flow, and algorithms.
   - **Identifier Expander**: Converts natural language concepts and code terms into:
     - `camelCase` (e.g., `preProcessInput`)
     - `PascalCase` (e.g., `PreProcessInput`)
     - `snake_case` (e.g., `pre_process_input`)
   - **StructuredQuery Schema**: Strict Pydantic model (`src.preprocessor.StructuredQuery`) providing typed contract for downstream sparse and dense retrieval roles.
   - **Zero Generative LLMs**: Purely deterministic, regex-based, and rule-based processing ensuring sub-millisecond query parsing with zero GPU or LLM generation overhead.

2. **MTEB Baseline Pipeline (`src/encoder.py`)**
   - **PrePostPipelineEncoder**: Subclasses `mteb.models.abs_encoder.AbsEncoder`.
   - **Model**: `sentence-transformers/all-MiniLM-L6-v2` loaded on CPU.
   - **Query Preprocessing Integration**: Intercepts queries (`prompt_type == PromptType.query`) through `QueryParser` to enrich dense representation with extracted tokens and intent prompts.
   - Directly conforms to MTEB 2.x interface and passes all test assertions.

3. **MTEB AppsRetrieval Evaluation (`evaluate.py`)**
   - Evaluated on the **CoIR `AppsRetrieval`** test split (8,765 corpus documents, 3,765 queries).
   - Generates the official `appsretrieval_results.json` artifact for submission.

---

## Baseline Results (CoIR AppsRetrieval)

| Metric | Score |
| :--- | :--- |
| **NDCG@10** | **0.05780** |
| **MRR@10** | **0.04829** |
| **MAP@10** | **0.04829** |
| **Recall@10** | **0.08871** |

---

## Repository Structure

```text
├── .gitignore
├── README.md
├── requirements.txt
├── evaluate.py
├── appsretrieval_results.json
├── src/
│   ├── __init__.py
│   ├── preprocessor.py   # Role 1: QueryParser, IdentifierExpander, StructuredQuery
│   └── encoder.py        # MTEB PrePostPipelineEncoder (AbsEncoder)
├── tests/
│   ├── test_preprocessor.py
│   └── test_encoder.py
└── data/
```

---

## Quickstart

### 1. Environment Setup
```bash
# Using uv (recommended) or standard venv
uv venv .venv
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

### 2. Run Unit Tests
```bash
python tests/test_preprocessor.py
python tests/test_encoder.py
```

### 3. Run MTEB Evaluation
```bash
python evaluate.py
```
This produces `appsretrieval_results.json` adhering to the MTEB evaluation format.
