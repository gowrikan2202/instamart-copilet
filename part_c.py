import pandas as pd
from mock_llm import est_tokens

MAX_INPUT_TOKENS = 400

def check_input_guardrails(row):
    text = row.get("text", "")
    
    # 1. Check if empty or only whitespace
    if not text or not text.strip():
        return "skipped_empty"
    
    # 2. Check if token count exceeds max limit
    tokens_count = est_tokens(text)
    if tokens_count > MAX_INPUT_TOKENS:
        return "flagged_long"
        
    return "passed"

if __name__ == "__main__":
    # Read tickets.csv with keep_default_na=False so empty rows remain empty strings
    df_tickets = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)

    statuses = []
    for idx, row in df_tickets.iterrows():
        status = check_input_guardrails(row)
        statuses.append(status)

    df_tickets["guardrail_status"] = statuses

    # Output-a terminal-la kaatturathukkaaka intha print statements add pannirukom:
    print("Total tickets processed:", len(df_tickets))
    print("\nGuardrail Status Counts:")
    print(df_tickets["guardrail_status"].value_counts())
    import re

def redact_text(text):
    if not isinstance(text, str):
        return text
    
    # 1. Redact Email
    text, email_count = re.subn(r'[\w.+-]+@[\w-]+\.[\w.]+', '[EMAIL]', text)
    
    # 2. Redact Phone
    text, phone_count = re.subn(r'\b[6-9]\d{9}\b', '[PHONE]', text)
    
    # 3. Redact Order ID
    text, order_count = re.subn(r'ORD-\d{6}', '[ORDER_ID]', text)
    
    return text, email_count, phone_count, order_count
# To test redaction across all tickets:
total_emails = 0
total_phones = 0
total_orders = 0

for text in df_tickets["text"]:
    redacted, e_cnt, p_cnt, o_cnt = redact_text(text)
    total_emails += e_cnt
    total_phones += p_cnt
    total_orders += o_cnt

print(f"Total Substitutions -> Emails: {total_emails}, Phones: {total_phones}, Order IDs: {total_orders}")
print(f"Total: {total_emails + total_phones + total_orders}") # Reference: 278
# Zero-Shot System Prompt
ZERO_SHOT_PROMPT = """CLASSIFY
You are a support ticket triage assistant for Swiggy Instamart.
Classify the given customer support ticket into one of the following allowed categories:
- billing
- delivery
- technical
- fraud

You must reply with a valid JSON object only, containing exactly two keys: "category" and "reason".
Format:
{"category": "<category_name>", "reason": "<short explanation>"}
"""

# Few-Shot System Prompt
FEW_SHOT_PROMPT = """CLASSIFY
You are a support ticket triage assistant for Swiggy Instamart.
Classify the given customer support ticket into one of the following allowed categories:
- billing
- delivery
- technical
- fraud

You must reply with a valid JSON object only, containing exactly two keys: "category" and "reason".
Format:
{"category": "<category_name>", "reason": "<short explanation>"}

EXAMPLES:
Ticket: "There is an unauthorized transaction on my card linked to order ORD-123456. I did not place it. Call 9876543210."
Output: {"category": "fraud", "reason": "Unauthorized transaction and card usage reported."}

Ticket: "My order ORD-654321 has not arrived and the rider is not picking up calls."
Output: {"category": "delivery", "reason": "Order delayed and rider unreachable."}
"""
import pandas as pd
import re
import random
import time
from mock_llm import est_tokens, chat, TransientLLMError, reset

MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
ALLOWED_CATS = {"billing", "delivery", "technical", "fraud"}

# 1. Guardrail function
def check_input_guardrails(row):
    text = row.get("text", "")
    if not text or not text.strip():
        return "skipped_empty"
    if est_tokens(text) > MAX_INPUT_TOKENS:
        return "flagged_long"
    return "passed"

# 2. Redaction function
def redact_text(text):
    if not isinstance(text, str):
        return text, 0, 0, 0
    text, e_cnt = re.subn(r'[\w.+-]+@[\w-]+\.[\w.]+', '[EMAIL]', text)
    text, p_cnt = re.subn(r'\b[6-9]\d{9}\b', '[PHONE]', text)
    text, o_cnt = re.subn(r'ORD-\d{6}', '[ORDER_ID]', text)
    return text, e_cnt, p_cnt, o_cnt

# 3. Prompts
ZERO_SHOT_PROMPT = """CLASSIFY
You are a support ticket triage assistant for Swiggy Instamart.
Classify the given customer support ticket into one of the following allowed categories:
- billing
- delivery
- technical
- fraud

You must reply with a valid JSON object only, containing exactly two keys: "category" and "reason".
Format:
{"category": "<category_name>", "reason": "<short explanation>"}
"""

FEW_SHOT_PROMPT = """CLASSIFY
You are a support ticket triage assistant for Swiggy Instamart.
Classify the given customer support ticket into one of the following allowed categories:
- billing
- delivery
- technical
- fraud

You must reply with a valid JSON object only, containing exactly two keys: "category" and "reason".
Format:
{"category": "<category_name>", "reason": "<short explanation>"}

EXAMPLES:
Ticket: "There is an unauthorized transaction on my card linked to order ORD-123456. I did not place it. Call 9876543210."
Output: {"category": "fraud", "reason": "Unauthorized transaction and card usage reported."}

Ticket: "My order ORD-654321 has not arrived and the rider is not picking up calls."
Output: {"category": "delivery", "reason": "Order delayed and rider unreachable."}
"""

# 4. Resilient LLM call with Retries
def call_llm_with_retry(messages):
    max_retries = 3
    for n in range(max_retries):
        try:
            return chat(messages)
        except TransientLLMError as e:
            wait = BASE_DELAY * (2 ** n) * random.uniform(0.9, 1.15)
            time.sleep(wait)
    raise Exception("Max retries exceeded for LLM call.")

# 5. Output Validation function
def validate_and_extract(response_content, original_text):
    import json
    # Try parsing JSON
    try:
        data = json.loads(response_content)
        cat = data.get("category")
        if cat in ALLOWED_CATS:
            return cat, "classified_json"
    except json.JSONDecodeError:
        pass
    
    # Keyword extraction fallback
    words = re.findall(r"[a-z]+", original_text.lower())
    matched_cats = [w for w in words if w in ALLOWED_CATS]
    if len(matched_cats) == 1:
        return matched_cats[0], "classified_keyword"
        
    return None, "flagged_invalid_output"

# Run Evaluation Pipeline
def run_pipeline(system_prompt):
    df = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)
    reset() # Reset mock llm state
    
    statuses = []
    categories = []
    total_tokens = 0
    
    json_count = 0
    keyword_count = 0
    invalid_count = 0
    
    correct_predictions = 0
    classified_total = 0
    
    for idx, row in df.iterrows():
        g_status = check_input_guardrails(row)
        if g_status != "passed":
            statuses.append(g_status)
            categories.append(None)
            continue
            
        redacted_text, _, _, _ = redact_text(row["text"])
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": redacted_text}
        ]
        
        try:
            resp = call_llm_with_retry(messages)
            total_tokens += resp.usage.total_tokens
            ans_content = resp.choices[0].message.content
            
            cat, val_status = validate_and_extract(ans_content, ans_content)
            
            if val_status == "classified_json":
                json_count += 1
            elif val_status == "classified_keyword":
                keyword_count += 1
            else:
                invalid_count += 1
                
            statuses.append(val_status)
            categories.append(cat)
            
            if cat and cat == row["true_category"]:
                correct_predictions += 1
            if val_status in ["classified_json", "classified_keyword"]:
                classified_total += 1
                
        except Exception as e:
            statuses.append("flagged_invalid_output")
            categories.append(None)
            invalid_count += 1
            
    df["assigned_status"] = statuses
    df["predicted_category"] = categories
    
    accuracy = (correct_predictions / classified_total) * 100 if classified_total > 0 else 0
    tokens_per_ticket = total_tokens / classified_total if classified_total > 0 else 0
    
    return {
        "df": df,
        "classified_total": classified_total,
        "json_count": json_count,
        "keyword_count": keyword_count,
        "invalid_count": invalid_count,
        "accuracy": accuracy,
        "total_tokens": total_tokens,
        "tokens_per_ticket": tokens_per_ticket
    }

if __name__ == "__main__":
    print("Running Zero-Shot Pipeline...")
    zs_results = run_pipeline(ZERO_SHOT_PROMPT)
    print(f"Zero-Shot -> Classified: {zs_results['classified_total']}, Accuracy: {zs_results['accuracy']:.2f}%, Total Tokens: {zs_results['total_tokens']}")
    print(f"Status breakdown -> JSON: {zs_results['json_count']}, Keyword: {zs_results['keyword_count']}, Invalid: {zs_results['invalid_count']}")
    
    print("\nRunning Few-Shot Pipeline...")
    fs_results = run_pipeline(FEW_SHOT_PROMPT)
    print(f"Few-Shot -> Classified: {fs_results['classified_total']}, Accuracy: {fs_results['accuracy']:.2f}%, Total Tokens: {fs_results['total_tokens']}")
    print(f"Status breakdown -> JSON: {fs_results['json_count']}, Keyword: {fs_results['keyword_count']}, Invalid: {fs_results['invalid_count']}")
    import pandas as pd
import re
import random
import time
from mock_llm import est_tokens, chat, TransientLLMError, reset

MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
ALLOWED_CATS = {"billing", "delivery", "technical", "fraud"}

# 1. Guardrail function
def check_input_guardrails(row):
    text = row.get("text", "")
    if not text or not text.strip():
        return "skipped_empty"
    if est_tokens(text) > MAX_INPUT_TOKENS:
        return "flagged_long"
    return "passed"

# 2. Redaction function
def redact_text(text):
    if not isinstance(text, str):
        return text, 0, 0, 0
    text, e_cnt = re.subn(r'[\w.+-]+@[\w-]+\.[\w.]+', '[EMAIL]', text)
    text, p_cnt = re.subn(r'\b[6-9]\d{9}\b', '[PHONE]', text)
    text, o_cnt = re.subn(r'ORD-\d{6}', '[ORDER_ID]', text)
    return text, e_cnt, p_cnt, o_cnt

# 3. Prompts
ZERO_SHOT_PROMPT = """CLASSIFY
You are a support ticket triage assistant for Swiggy Instamart.
Classify the given customer support ticket into one of the following allowed categories:
- billing
- delivery
- technical
- fraud

You must reply with a valid JSON object only, containing exactly two keys: "category" and "reason".
Format:
{"category": "<category_name>", "reason": "<short explanation>"}
"""

FEW_SHOT_PROMPT = """CLASSIFY
You are a support ticket triage assistant for Swiggy Instamart.
Classify the given customer support ticket into one of the following allowed categories:
- billing
- delivery
- technical
- fraud

You must reply with a valid JSON object only, containing exactly two keys: "category" and "reason".
Format:
{"category": "<category_name>", "reason": "<short explanation>"}

EXAMPLES:
Ticket: "There is an unauthorized transaction on my card linked to order ORD-123456. I did not place it. Call 9876543210."
Output: {"category": "fraud", "reason": "Unauthorized transaction and card usage reported."}

Ticket: "My order ORD-654321 has not arrived and the rider is not picking up calls."
Output: {"category": "delivery", "reason": "Order delayed and rider unreachable."}
"""

# 4. Resilient LLM call with Retries
def call_llm_with_retry(messages):
    max_retries = 3
    for n in range(max_retries):
        try:
            return chat(messages)
        except TransientLLMError as e:
            wait = BASE_DELAY * (2 ** n) * random.uniform(0.9, 1.15)
            time.sleep(wait)
    raise Exception("Max retries exceeded for LLM call.")

# 5. Output Validation function
def validate_and_extract(response_content, original_text):
    import json
    try:
        data = json.loads(response_content)
        cat = data.get("category")
        if cat in ALLOWED_CATS:
            return cat, "classified_json"
    except json.JSONDecodeError:
        pass
    
    words = re.findall(r"[a-z]+", original_text.lower())
    matched_cats = [w for w in words if w in ALLOWED_CATS]
    if len(matched_cats) == 1:
        return matched_cats[0], "classified_keyword"
        
    return None, "flagged_invalid_output"

# 6. Pipeline Runner with Token Budget Support
def run_pipeline_with_budget(system_prompt, token_budget):
    df = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)
    reset()
    
    statuses = []
    categories = []
    tokens_used_list = []
    running_tokens = 0
    
    for idx, row in df.iterrows():
        g_status = check_input_guardrails(row)
        if g_status != "passed":
            statuses.append(g_status)
            categories.append(None)
            tokens_used_list.append(0)
            continue
            
        # Check token budget before model call
        if running_tokens >= token_budget:
            statuses.append("budget_exhausted")
            categories.append(None)
            tokens_used_list.append(0)
            continue
            
        redacted_text, _, _, _ = redact_text(row["text"])
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": redacted_text}
        ]
        
        try:
            resp = call_llm_with_retry(messages)
            t_tokens = resp.usage.total_tokens
            
            # Check if adding this response exceeds budget
            if running_tokens + t_tokens > token_budget:
                statuses.append("budget_exhausted")
                categories.append(None)
                tokens_used_list.append(0)
                continue
                
            running_tokens += t_tokens
            tokens_used_list.append(t_tokens)
            
            ans_content = resp.choices[0].message.content
            cat, val_status = validate_and_extract(ans_content, ans_content)
            
            statuses.append(val_status)
            categories.append(cat)
            
        except Exception:
            statuses.append("flagged_invalid_output")
            categories.append(None)
            tokens_used_list.append(0)
            
    df["status"] = statuses
    df["category"] = categories
    df["tokens_used"] = tokens_used_list
    return df

if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)
    
    print("Running Step 7a: Budget Demo (Zero-Shot, Budget = 6000)...")
    df_budget = run_pipeline_with_budget(ZERO_SHOT_PROMPT, token_budget=6000)
    df_budget.to_csv("outputs/triage_budget_demo.csv", index=False)
    exhausted_count = (df_budget["status"] == "budget_exhausted").sum()
    print(f"Budget Demo completed. Tickets with 'budget_exhausted': {exhausted_count}")
    
    print("\nRunning Step 7b: Final Triage Results (Few-Shot, Budget = 100000)...")
    df_final = run_pipeline_with_budget(FEW_SHOT_PROMPT, token_budget=100000)
    
    # Load customer segments from Part B
    try:
        df_segments = pd.read_csv("outputs/customer_segments.csv")
        premium_map = dict(zip(df_segments["customer_id"], df_segments["is_premium"]))
        segment_name_map = dict(zip(df_segments["customer_id"], df_segments["segment_name"]))
    except FileNotFoundError:
        premium_map = {}
        segment_name_map = {}
        
    routes = []
    priorities = []
    segment_names = []
    
    for idx, row in df_final.iterrows():
        status = row["status"]
        cat = row["category"]
        cid = row["customer_id"]
        
        seg_name = segment_name_map.get(cid, "")
        is_prem = premium_map.get(cid, False)
        
        segment_names.append(seg_name if pd.notna(cat) and cat else "")
        
        # Routing Logic
        if status in ["skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"]:
            routes.append("human_review")
            priorities.append("NORMAL")
        elif cat == "fraud":
            routes.append("security_team")
            priorities.append("HIGH")
        elif is_prem:
            routes.append("premium_desk")
            priorities.append("HIGH")
        else:
            routes.append("support_queue")
            priorities.append("NORMAL")
            
    df_final["segment_name"] = segment_names
    df_final["priority"] = priorities
    df_final["route"] = routes
    
    # Select final requested columns
    cols_to_save = ["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"]
    df_final_out = df_final[cols_to_save]
    
    # Clean up empty category values if any
    df_final_out.loc[df_final_out["category"].isna(), "category"] = ""
    df_final_out.loc[df_final_out["segment_name"].isna(), "segment_name"] = ""
    
    df_final_out.to_csv("outputs/triage_results.csv", index=False)
    print("Final triage results saved to outputs/triage_results.csv successfully!")
    print(df_final_out["route"].value_counts())