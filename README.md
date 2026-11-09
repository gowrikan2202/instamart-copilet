# Swiggy Instamart Customer Intelligence & Support Copilot

## Overview
- **Project Scope:** An end-to-end data engineering, machine learning, and AI agent pipeline built for Swiggy Instamart across four connected parts.
- **Components:** 
  1. Data Foundation (Cleaning, SQLite storage, and SQL queries)
  2. Customer Segments (RFM analysis & K-means clustering) & Lapse Prediction Model
  3. Support Ticket Triage Pipeline (Privacy guardrails, validation, resilient LLM calls)
  4. Policy RAG Assistant & ReAct Refund Agent (Stateful, resumable operations)

---

## Reproduce
To replicate this entire project from scratch in a clean environment, run the following shell commands in order:

# 1. Create and activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate

# 2. Install required dependencies
pip install -r requirements.txt

# 3. Generate raw datasets and policies
python generate_data.py

# 4. Run Part A: Data Pipeline & Validation
python part_a.py

# 5. Run Part B: Customer Segmentation & Lapse Model
python part_b.py

# 6. Run Part C: Support Ticket Triage Pipeline
python part_c.py

# 7. Run Part D: Policy Assistant & Refund Agent Scenarios
python part_d.py

## Part A  
* Data Foundation:Clean, Store, Query
- Raw Data Audit:customers.csv: 
   Shape (1200, 5) | Missing: age 61 (5.08%), gender 30 (2.50%)
- products.csv:
  Shape (40, 4) | Missing:0
- orders.csv: Shape (13084, 9) | 119 exact duplicates, 656 pending  rows,   12,309 distinct    order  IDs, 67 unparseable amounts, 106 negative delivery times, 2,541 blank delivery times.
- tickets.csv: Shape (164, 5) | 0 missing values.
* Cleaning Summary & Rules:
- Customers: Cities standardized by stripping spaces and applying title case to 6 metro regions (Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184). Genders mapped to male (560), female (610), or unknown (30 for blank). Ages outside 10–100 treated as missing and imputed using city-wise median ages.
- Orders: Converted timestamps to datetime; removed exact duplicates; kept latest record per order_id for multi-status rows; imputed unparseable amounts using formula price_inr * quantity * (1 
- discount_pct / 100); set negative delivery times to missing and imputed completed orders' missing delivery times using category-wise median delivery minutes, leaving cancelled orders' delivery times missing.
- Final Clean Orders: 12,309 rows (1,1110 completed, 1,199 cancelled); sum of completed order_amount is INR 2,998,668.80.
* SQL Analysis Results:
- Q1: 30 customers.
- Q2: 30 rows; total revenue INR 2,998,668.80; largest cell Bengaluru / Electronics at INR 233,909.45.
- Q3: 11 products.
- Q4: 25 customers.
- Q5: 550 customers; average spend INR 2,552.06.
- Q6: Top customers per city: Bengaluru C0082 (12,856.20), Chennai C0930 (10,151.40), Delhi C0291 (13,742.60), Hyderabad C0743 (12,730.05), Mumbai C0693 (14,766.55), Pune C0426 (10,377.10). Exactly 8 rows differ in rank vs row number due to exact ties in total spend.
Q7: 183 daily rows; final cumulative revenue INR 2,998,668.80.
## Part B — Customer Segments and Lapse Prediction
* RFM Analysis: 1,175 customers analyzed as of 2025-10-01 00:00:00. Recency median 19 days, frequency mean 9.46, monetary mean INR 2,552.06.
* K-Means Clustering: Chosen $K=4$ based on silhouette score (0.526) and business interpretability.
* Segment Profiles:
- Segment 0 (Low-Value / Occasional): Size 436, Mean Recency 28.86, Frequency 3.77, Monetary INR 1,057.62.
- Segment 1 (High-Value / Premium): Size 138, Mean Recency 24.12, Frequency 6.77, Monetary INR 7,129.70 (Marked as Premium).
- Segment 2 (Loyal / Frequent): Size 419, Mean Recency 8.77, Frequency 19.40, Monetary INR 3,410. 
- Segment 3 (At-Risk / Lapsed): Size 182, Mean Recency 109.04, Frequency 2.20, Monetary INR 685.99.
* PCA Ratios: Explained variance ratio: [0.6343, 0.2180] (Total ~85.2%).
* Lapse Model Performance: 1,156 customers in cohort; 443 lapsed (38.32%). Baseline accuracy 61.6% (F1 = 0). Tuned Random Forest test F1 score: 0.6448 (exceeds the 0.55 requirement).
* Outputs Saved: outputs/customer_segments.csv, outputs/lapse_model.joblib, outputs/retention_list.csv.
## Part C — Support Ticket Triage with Guardrails
* Input Guardrails: 4 empty tickets (skipped_empty) and 4 extremely long tickets (flagged_long) filtered out.
* PII Masking: Sanitized across 164 tickets: 55 emails, 91 phone numbers, 132 order IDs (Total: 278 substitutions). Zero PII remained after masking.
* Model Execution & Validation: Full run produced 19 TransientLLMErrors, all successfully recovered via exponential backoff retry. Status counts: 135 classified_json, 12 classified_keyword, 9 flagged_invalid_output.
* Comparison: Zero-shot accuracy: 85.71%; Few-shot accuracy: 100.00%. Few-shot prompt tokens per ticket are higher, but worth paying for to eliminate classification errors on fraud and technical tickets.
* Budget Demo: Zero-shot with TOKEN_BUDGET = 6000 exhausted after processing tickets (96 tickets marked budget_exhausted). Final production run saved to outputs/triage_results.csv.
## Part D — Policy RAG Assistant and Refund Agent
* Chunking & Retrieval: 15 chunks created (40-word chunks with 10-word overlap). TF-IDF retrieval evaluated successfully on 14 evaluation questions (12/12 expected documents in top 3, 12/12 expected facts found, 2/2 unanswerable queries correctly caught).
* Safeties & Resumability Demonstrations:

  - S1 with token_limit=200 ended with status FALLBACK_TOKEN_LIMIT.

  - S1 with time_limit_seconds=0 ended with status FALLBACK_TIMEOUT.

  - S7 under default limits ended with status FALLBACK_MAX_ITERATIONS after 6 model calls.

* Crash-and-Resume Run: Simulated crash after the 2nd tool execution for S1 successfully resumed using resume_agent. Tools executed in order: get_order, search_policy, calculate_refund, issue_refund (4 executions, no repetitions, final status REFUND with INR 128.25).

## Limitations and Next Steps
* Limitations:

- The Premium segment flag used in Part B's retention list is a post-hoc whole-window label (incorporating September activity), meaning the retention list is illustrative rather than leakage-free.

 - TF-IDF retrieval relies purely on lexical keyword matching, which can struggle with semantic synonyms or paraphrased queries.

 * Next Steps: Integrate dense embedding models (vector search) for policy retrieval, and deploy the refund copilot with live API endpoints and asynchronous queue management.