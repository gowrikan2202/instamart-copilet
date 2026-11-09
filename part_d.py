import os
import glob

POL_DIR = "data/policies"

def chunk_documents():
    chunks = []
    # Find all text files in policies directory
    for filepath in sorted(glob.glob(os.path.join(POL_DIR, "*.txt"))):
        filename = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
            
        words = text.split()
        chunk_size = 40
        overlap = 10
        step = chunk_size - overlap  # 30 words step
        
        i = 0
        while i < len(words):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            chunks.append({
                "source": filename,
                "text": chunk_text
            })
            # Stop after the first chunk whose end reaches or passes the last word
            if i + chunk_size >= len(words):
                break
            i += step
            
    return chunks

if __name__ == "__main__":
    all_chunks = chunk_documents()
    print(f"Total chunks created: {len(all_chunks)}") # Reference: 15 chunks
    for c in all_chunks[:3]:
        print(f"[{c['source']}] {c['text'][:60]}...")
        from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Global vectorizer and corpus structures
vectorizer = None
chunk_corpus = []
tfidf_matrix = None

def prepare_retrieval(chunks):
    global vectorizer, chunk_corpus, tfidf_matrix
    chunk_corpus = chunks
    texts = [c["text"] for c in chunks]
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)

def retrieve(query, k=3):
    global vectorizer, chunk_corpus, tfidf_matrix
    if vectorizer is None:
        raise ValueError("Retrieval system not initialized. Call prepare_retrieval first.")
        
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Get top k indices sorted by similarity score descending
    top_indices = np.argsort(similarities)[::-1][:k]
    
    results = []
    for idx in top_indices:
        results.append({
            "source": chunk_corpus[idx]["source"],
            "score": float(similarities[idx]),
            "text": chunk_corpus[idx]["text"]
        })
    return results
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Global vectorizer and corpus structures
vectorizer = None
chunk_corpus = []
tfidf_matrix = None

def prepare_retrieval(chunks):
    global vectorizer, chunk_corpus, tfidf_matrix
    chunk_corpus = chunks
    texts = [c["text"] for c in chunks]
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)

def retrieve(query, k=3):
    global vectorizer, chunk_corpus, tfidf_matrix
    if vectorizer is None:
        raise ValueError("Retrieval system not initialized. Call prepare_retrieval first.")
        
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Get top k indices sorted by similarity score descending
    top_indices = np.argsort(similarities)[::-1][:k]
    
    results = []
    for idx in top_indices:
        results.append({
            "source": chunk_corpus[idx]["source"],
            "score": float(similarities[idx]),
            "text": chunk_corpus[idx]["text"]
        })
    return results
if __name__ == "__main__":
    all_chunks = chunk_documents()
    print(f"Total chunks created: {len(all_chunks)}")
    
    prepare_retrieval(all_chunks)
    
    test_query = "refund policy late delivery"
    top_results = retrieve(test_query, k=3)
    
    print(f"\nTop 3 chunks for query: '{test_query}'")
    for r in top_results:
        print(f"Source: {r['source']} | Score: {r['score']:.4f}")
        print(f"Text: {r['text'][:80]}...\n")
        from mock_llm import chat

def answer(question):
    # 1. Retrieve top 3 chunks
    top_chunks = retrieve(question, k=3)
    
    # 2. Build context string with source file names
    context_lines = []
    sources_used = set()
    for chunk in top_chunks:
        src = chunk["source"]
        sources_used.add(src)
        context_lines.append(f"[{src}]\n{chunk['text']}")
    
    context_str = "\n\n".join(context_lines)
    
    # 3. Construct user message strictly matching project format
    user_msg = f"CONTEXT:\n{context_str}\n\nQUESTION:\n{question}"
    
    # 4. Construct messages with required system prompt marker
    messages = [
        {"role": "system", "content": "POLICY ASSISTANT: Answer only from the supplied context. If the answer is not present, say 'I could not find that information in the policy documents.'"},
        {"role": "user", "content": user_msg}
    ]
    
    # 5. Call LLM
    response = chat(messages)
    answer_text = response.choices[0].message.content
    
    return answer_text, list(sources_used)
if __name__ == "__main__":
    all_chunks = chunk_documents()
    prepare_retrieval(all_chunks)
    
    q = "What delivery time makes a completed order eligible for a full refund?"
    ans, srcs = answer(q)
    print(f"\nQuestion: {q}")
    print(f"Answer: {ans}")
    print(f"Sources: {srcs}")
    import csv
import os

def evaluate_policy_rag(zero_overlap=False):
    eval_path = os.path.join("data", "raw", "policy_eval.csv")
    if not os.path.exists(eval_path):
        print("policy_eval.csv not found!")
        return

    # Re-chunk documents if zero_overlap is True
    chunks = chunk_documents_custom(zero_overlap=zero_overlap)
    prepare_retrieval(chunks)

    doc_matches = 0
    fact_matches = 0
    fallback_matches = 0
    answerable_count = 0
    unanswerable_count = 0

    with open(eval_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row["question"]
            exp_doc = row["expected_doc"]
            exp_fact = row["expected_fact"]

            ans, srcs = answer(q)

            if exp_doc == "NONE":
                unanswerable_count += 1
                if ans.startswith("I could not find"):
                    fallback_matches += 1
            else:
                answerable_count += 1
                # Check if expected doc is in top 3 retrieved sources
                if exp_doc in srcs:
                    doc_matches += 1
                # Check if expected fact is a case-sensitive substring in answer
                if exp_fact in ans:
                    fact_matches += 1

    print(f"\n--- Evaluation Results (Zero Overlap = {zero_overlap}) ---")
    print(f"Answerable Questions ({answerable_count}):")
    print(f"  - Expected Doc in Top 3: {doc_matches}/{answerable_count}")
    print(f"  - Expected Fact in Answer: {fact_matches}/{answerable_count}")
    print(f"Unanswerable Questions ({unanswerable_count}):")
    print(f"  - Correct Fallback Response: {fallback_matches}/{unanswerable_count}")

# Helper for custom chunking with configurable overlap
def chunk_documents_custom(zero_overlap=False):
    chunks = []
    pol_dir = "data/policies"
    for filepath in sorted(glob.glob(os.path.join(pol_dir, "*.txt"))):
        filename = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
            
        words = text.split()
        chunk_size = 40
        overlap = 0 if zero_overlap else 10
        step = chunk_size - overlap
        
        i = 0
        while i < len(words):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            chunks.append({"source": filename, "text": chunk_text})
            if i + chunk_size >= len(words):
                break
            i += step
    return chunks
if __name__ == "__main__":
    all_chunks = chunk_documents()
    prepare_retrieval(all_chunks)
    
    # Run evaluation with default overlap (10)
    evaluate_policy_rag(zero_overlap=False)
    
    # Run evaluation with zero overlap (0)
    evaluate_policy_rag(zero_overlap=True)
    import sqlite3
import json
from datetime import datetime

DB_PATH = "data/instamart.db"

# 1. Individual Tool Functions
def get_order(order_id):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?", (order_id,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return {"error": "order not found"}
        
    return {
        "order_id": row["order_id"],
        "status": row["status"],
        "order_amount": float(row["order_amount"]) if row["order_amount"] is not None else 0.0,
        "delivery_minutes": int(row["delivery_minutes"]) if row["delivery_minutes"] is not None and row["delivery_minutes"] != "" else 0
    }

def search_policy(query):
    top_chunks = retrieve(query, k=3)
    return {"results": top_chunks}

def calculate_refund(order_amount, delivery_minutes):
    # Refund rules:
    # - Only completed orders (handled by agent flow/tool order)
    # - Delivered in > 45 mins -> 100% refund
    # - Delivered in <= 45 mins -> 0 refund
    if delivery_minutes > 45:
        return {"refund_inr": float(order_amount)}
    else:
        return {"refund_inr": 0.0}

def issue_refund(order_id, refund_inr):
    os.makedirs("outputs", exist_ok=True)
    ledger_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tool": "issue_refund",
        "order_id": order_id,
        "refund_inr": float(refund_inr)
    }
    with open(os.path.join("outputs", "refund_ledger.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(ledger_entry) + "\n")
    return {"refund_inr": float(refund_inr)}

def escalate_to_human(order_id, refund_inr):
    os.makedirs("outputs", exist_ok=True)
    ledger_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tool": "escalate_to_human",
        "order_id": order_id,
        "refund_inr": float(refund_inr)
    }
    with open(os.path.join("outputs", "refund_ledger.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(ledger_entry) + "\n")
    return {"refund_inr": float(refund_inr)}

# 2. TOOL_MAP dictionary
TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}

# 3. TOOLS_SPEC list (JSON-style parameter specification for agent)
TOOLS_SPEC = [
    {
        "name": "get_order",
        "description": "Looks up an order ID in the database and returns its status, amount, and delivery time.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"]
        }
    },
    {
        "name": "search_policy",
        "description": "Searches company policy documents using a text query and returns relevant chunks.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"]
        }
    },
    {
        "name": "calculate_refund",
        "description": "Calculates the refund amount based on order amount and delivery minutes.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_amount": {"type": "number"},
                "delivery_minutes": {"type": "integer"}
            },
            "required": ["order_amount", "delivery_minutes"]
        }
    },
    {
        "name": "issue_refund",
        "description": "Issues an automated refund up to INR 1000 and records it in the ledger.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"}
            },
            "required": ["order_id", "refund_inr"]
        }
    },
    {
        "name": "escalate_to_human",
        "description": "Escalates a refund request above INR 1000 to a human lead and logs it.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"}
            },
            "required": ["order_id", "refund_inr"]
        }
    }
]
import os
import glob
import csv
import json
import time
import sqlite3
import numpy as np
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from mock_llm import chat, reset

POL_DIR = "data/policies"
DB_PATH = "data/instamart.db"

# --- TASK 1: Chunking ---
def chunk_documents():
    chunks = []
    for filepath in sorted(glob.glob(os.path.join(POL_DIR, "*.txt"))):
        filename = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
            
        words = text.split()
        chunk_size = 40
        overlap = 10
        step = chunk_size - overlap
        
        i = 0
        while i < len(words):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            chunks.append({"source": filename, "text": chunk_text})
            if i + chunk_size >= len(words):
                break
            i += step
    return chunks

# --- TASK 2: Vectorise & Retrieve ---
vectorizer = None
chunk_corpus = []
tfidf_matrix = None

def prepare_retrieval(chunks):
    global vectorizer, chunk_corpus, tfidf_matrix
    chunk_corpus = chunks
    texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)

def retrieve(query, k=3):
    global vectorizer, chunk_corpus, tfidf_matrix
    if vectorizer is None:
        raise ValueError("Retrieval system not initialized.")
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = np.argsort(similarities)[::-1][:k]
    
    results = []
    for idx in top_indices:
        results.append({
            "source": chunk_corpus[idx]["source"],
            "score": float(similarities[idx]),
            "text": chunk_corpus[idx]["text"]
        })
    return results

# --- TASK 3: Augment & Answer ---
def answer(question):
    top_chunks = retrieve(question, k=3)
    context_lines = []
    sources_used = set()
    for chunk in top_chunks:
        src = chunk["source"]
        sources_used.add(src)
        context_lines.append(f"[{src}]\n{chunk['text']}")
    
    context_str = "\n\n".join(context_lines)
    user_msg = f"CONTEXT:\n{context_str}\n\nQUESTION:\n{question}"
    
    messages = [
        {"role": "system", "content": "POLICY ASSISTANT: Answer only from the supplied context. If the answer is not present, say 'I could not find that information in the policy documents.'"},
        {"role": "user", "content": user_msg}
    ]
    response = chat(messages)
    return response.choices[0].message.content, list(sources_used)

# --- TASK 5: Tools & Tool Map ---
def get_order(order_id):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?", (order_id,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return {"error": "order not found"}
        
    return {
        "order_id": row["order_id"],
        "status": row["status"],
        "order_amount": float(row["order_amount"]) if row["order_amount"] is not None else 0.0,
        "delivery_minutes": int(row["delivery_minutes"]) if row["delivery_minutes"] is not None and row["delivery_minutes"] != "" else 0
    }

def search_policy(query):
    top_chunks = retrieve(query, k=3)
    return {"results": top_chunks}

def calculate_refund(order_amount, delivery_minutes):
    if delivery_minutes > 45:
        return {"refund_inr": float(order_amount)}
    else:
        return {"refund_inr": 0.0}

def issue_refund(order_id, refund_inr):
    os.makedirs("outputs", exist_ok=True)
    ledger_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tool": "issue_refund",
        "order_id": order_id,
        "refund_inr": float(refund_inr)
    }
    with open(os.path.join("outputs", "refund_ledger.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(ledger_entry) + "\n")
    return {"refund_inr": float(refund_inr)}

def escalate_to_human(order_id, refund_inr):
    os.makedirs("outputs", exist_ok=True)
    ledger_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tool": "escalate_to_human",
        "order_id": order_id,
        "refund_inr": float(refund_inr)
    }
    with open(os.path.join("outputs", "refund_ledger.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(ledger_entry) + "\n")
    return {"refund_inr": float(refund_inr)}

TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}

TOOLS_SPEC = [
    {
        "name": "get_order",
        "description": "Looks up an order ID in the database and returns its status, amount, and delivery time.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"]
        }
    },
    {
        "name": "search_policy",
        "description": "Searches company policy documents using a text query and returns relevant chunks.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"]
        }
    },
    {
        "name": "calculate_refund",
        "description": "Calculates the refund amount based on order amount and delivery minutes.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_amount": {"type": "number"},
                "delivery_minutes": {"type": "integer"}
            },
            "required": ["order_amount", "delivery_minutes"]
        }
    },
    {
        "name": "issue_refund",
        "description": "Issues an automated refund up to INR 1000 and records it in the ledger.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"}
            },
            "required": ["order_id", "refund_inr"]
        }
    },
    {
        "name": "escalate_to_human",
        "description": "Escalates a refund request above INR 1000 to a human lead and logs it.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"}
            },
            "required": ["order_id", "refund_inr"]
        }
    }
]

# --- TASK 6: ReAct Loop & Agent Execution ---
def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10):
    start_time = time.monotonic()
    reset()
    
    messages = [
        {"role": "system", "content": "AGENT: You are an autonomous refund assistant for Swiggy Instamart. Use the supplied tools to resolve the customer's request safely and correctly."},
        {"role": "user", "content": user_message}
    ]
    
    tools_executed = []
    steps = []
    llm_calls = 0
    tokens_used = 0
    
    for step_num in range(1, max_iterations + 1):
        if (time.monotonic() - start_time) >= time_limit_seconds:
            return {
                "scenario_id": scenario_id,
                "status": "FALLBACK_TIMEOUT",
                "final_answer": "FALLBACK: Timeout limit reached.",
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
                "tools_executed": tools_executed
            }
            
        try:
            response = chat(messages, tools=TOOLS_SPEC)
            llm_calls += 1
            tokens_used += response.usage.total_tokens
        except Exception as e:
            return {
                "scenario_id": scenario_id,
                "status": "FALLBACK_ERROR",
                "final_answer": f"FALLBACK: {str(e)}",
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
                "tools_executed": tools_executed
            }
            
        if tokens_used > token_limit:
            return {
                "scenario_id": scenario_id,
                "status": "FALLBACK_TOKEN_LIMIT",
                "final_answer": "FALLBACK: Token budget exhausted.",
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
                "tools_executed": tools_executed
            }
            
        message = response.choices[0].message
        
        if not message.tool_calls or len(message.tool_calls) == 0:
            final_content = message.content or ""
            status_word = final_content.split(":")[0].strip() if ":" in final_content else "UNKNOWN"
            return {
                "scenario_id": scenario_id,
                "status": status_word,
                "final_answer": final_content,
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
                "tools_executed": tools_executed
            }
            
        tool_calls_raw = []
        for tc in message.tool_calls:
            args = tc.arguments
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            tool_calls_raw.append({
                "id": tc.id,
                "name": tc.name,
                "arguments": args
            })
            
        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": tool_calls_raw
        })
        
        for tc in tool_calls_raw:
            t_name = tc["name"]
            t_args = tc["arguments"]
            tools_executed.append(t_name)
            
            if t_name in TOOL_MAP:
                try:
                    obs = TOOL_MAP[t_name](**t_args)
                except Exception as ex:
                    obs = {"error": str(ex)}
            else:
                obs = {"error": f"Tool {t_name} not found"}
                
            obs_json = json.dumps(obs)
            steps.append({
                "step": step_num,
                "tool": t_name,
                "arguments": t_args,
                "observation": obs
            })
            
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "name": t_name,
                "content": obs_json
            })
            
    return {
        "scenario_id": scenario_id,
        "status": "FALLBACK_MAX_ITERATIONS",
        "final_answer": "FALLBACK: Max iterations reached.",
        "steps": steps,
        "llm_calls": llm_calls,
        "tokens_used": tokens_used,
        "tools_executed": tools_executed
    }

def run_all_scenarios():
    eval_path = os.path.join("data", "raw", "agent_scenarios.csv")
    if not os.path.exists(eval_path):
        print("agent_scenarios.csv not found!")
        return
        
    os.makedirs("outputs", exist_ok=True)
    ledger_path = os.path.join("outputs", "agent_runs.jsonl")
    if os.path.exists(ledger_path):
        os.remove(ledger_path)
        
    with open(eval_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sc_id = row["scenario_id"]
            msg = row["message"]
            res = run_agent(msg, sc_id, max_iterations=6, token_limit=6000, time_limit_seconds=10)
            
            with open(ledger_path, "a", encoding="utf-8") as lf:
                lf.write(json.dumps(res) + "\n")
            print(f"Scenario {sc_id} -> Status: {res['status']}, Calls: {res['llm_calls']}, Tokens: {res['tokens_used']}")

if __name__ == "__main__":
    chunks = chunk_documents()
    prepare_retrieval(chunks)
    print("Running Agent Scenarios...")
    run_all_scenarios()
def save_checkpoint(run_id, state_data):
    os.makedirs("state", exist_ok=True)
    cp_path = os.path.join("state", f"{run_id}.json")
    with open(cp_path, "w", encoding="utf-8") as f:
        json.dump(state_data, f, indent=2)

def resume_agent(run_id, max_iterations=6, token_limit=6000, time_limit_seconds=10):
    cp_path = os.path.join("state", f"{run_id}.json")
    if not os.path.exists(cp_path):
        return {"error": "checkpoint not found"}
        
    with open(cp_path, "r", encoding="utf-8") as f:
        state = json.load(f)
        
    scenario_id = state["scenario_id"]
    messages = state["messages"]
    steps = state["steps"]
    tools_executed = state["tools_executed"]
    tokens_used = state["tokens_used"]
    start_step = state["step_number"] + 1
    
    start_time = time.monotonic()
    
    for step_num in range(start_step, max_iterations + 1):
        if (time.monotonic() - start_time) >= time_limit_seconds:
            return {"scenario_id": scenario_id, "status": "FALLBACK_TIMEOUT", "tools_executed": tools_executed}
            
        try:
            response = chat(messages, tools=TOOLS_SPEC)
            tokens_used += response.usage.total_tokens
        except Exception as e:
            return {"scenario_id": scenario_id, "status": "FALLBACK_ERROR", "tools_executed": tools_executed}
            
        if tokens_used > token_limit:
            return {"scenario_id": scenario_id, "status": "FALLBACK_TOKEN_LIMIT", "tools_executed": tools_executed}
            
        message = response.choices[0].message
        
        if not message.tool_calls or len(message.tool_calls) == 0:
            final_content = message.content or ""
            status_word = final_content.split(":")[0].strip() if ":" in final_content else "UNKNOWN"
            return {
                "scenario_id": scenario_id,
                "status": status_word,
                "final_answer": final_content,
                "steps": steps,
                "tools_executed": tools_executed
            }
            
        tool_calls_raw = []
        for tc in message.tool_calls:
            args = tc.arguments
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            tool_calls_raw.append({"id": tc.id, "name": tc.name, "arguments": args})
            
        messages.append({"role": "assistant", "content": message.content, "tool_calls": tool_calls_raw})
        
        for tc in tool_calls_raw:
            t_name = tc["name"]
            t_args = tc["arguments"]
            tools_executed.append(t_name)
            
            if t_name in TOOL_MAP:
                try:
                    obs = TOOL_MAP[t_name](**t_args)
                except Exception as ex:
                    obs = {"error": str(ex)}
            else:
                obs = {"error": f"Tool {t_name} not found"}
                
            obs_json = json.dumps(obs)
            steps.append({"step": step_num, "tool": t_name, "arguments": t_args, "observation": obs})
            messages.append({"role": "tool", "tool_call_id": tc["id"], "name": t_name, "content": obs_json})
            
    return {"scenario_id": scenario_id, "status": "FALLBACK_MAX_ITERATIONS", "tools_executed": tools_executed}
def demonstrate_crash_and_resume():
    print("\n--- Running Crash & Resume Demo for S1 ---::")
    run_id = "s1_crash_demo"
    user_message = "My order ORD-100002 reached me very late. I would like a refund." # S1 message approx or fetch from csv
    # Let's read S1 message directly from agent_scenarios.csv
    eval_path = os.path.join("data", "raw", "agent_scenarios.csv")
    with open(eval_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        row = next(reader) # S1 is first row
        user_message = row["message"]
        sc_id = row["scenario_id"]
        
    reset()
    messages = [
        {"role": "system", "content": "AGENT: You are an autonomous refund assistant for Swiggy Instamart."},
        {"role": "user", "content": user_message}
    ]
    
    # Step 1: First LLM call & get_order tool execution
    response = chat(messages, tools=TOOLS_SPEC)
    message = response.choices[0].message
    tc = message.tool_calls[0]
    args = json.loads(tc.arguments) if isinstance(tc.arguments, str) else tc.arguments
    
    obs = TOOL_MAP[tc.name](**args)
    tools_executed = [tc.name]
    steps = [{"step": 1, "tool": tc.name, "arguments": args, "observation": obs}]
    
    messages.append({"role": "assistant", "content": message.content, "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": args}]})
    messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": json.dumps(obs)})
    
    # Step 2: Second LLM call & search_policy tool execution
    response2 = chat(messages, tools=TOOLS_SPEC)
    message2 = response2.choices[0].message
    tc2 = message2.tool_calls[0]
    args2 = json.loads(tc2.arguments) if isinstance(tc2.arguments, str) else tc2.arguments
    
    obs2 = TOOL_MAP[tc2.name](**args2)
    tools_executed.append(tc2.name)
    steps.append({"step": 2, "tool": tc2.name, "arguments": args2, "observation": obs2})
    
    messages.append({"role": "assistant", "content": message2.content, "tool_calls": [{"id": tc2.id, "name": tc2.name, "arguments": args2}]})
    messages.append({"role": "tool", "tool_call_id": tc2.id, "name": tc2.name, "content": json.dumps(obs2)})
    
    # SAVE CHECKPOINT RIGHT AFTER 2ND TOOL EXECUTION (As required by Task 8)
    state_data = {
        "run_id": run_id,
        "scenario_id": sc_id,
        "messages": messages,
        "steps": steps,
        "tools_executed": tools_executed,
        "tokens_used": response.usage.total_tokens + response2.usage.total_tokens,
        "step_number": 2
    }
    save_checkpoint(run_id, state_data)
    print("Simulated crash triggered after 2nd tool execution. Checkpoint saved.")
    
    # RESUME AGENT
    print("Resuming agent from checkpoint...")
    final_res = resume_agent(run_id)
    print(f"Resumed Final Status: {final_res['status']}")
    print(f"Tools Executed Across Crash & Resume: {final_res['tools_executed']}")
# --- TASK 8: Checkpoints & Resume Agent ---
def save_checkpoint(run_id, state_data):
    os.makedirs("state", exist_ok=True)
    cp_path = os.path.join("state", f"{run_id}.json")
    with open(cp_path, "w", encoding="utf-8") as f:
        json.dump(state_data, f, indent=2)
def resume_agent(run_id, max_iterations=6, token_limit=6000, time_limit_seconds=10):
    cp_path = os.path.join("state", f"{run_id}.json")
    if not os.path.exists(cp_path):
        return {"error": "checkpoint not found"}
        
    with open(cp_path, "r", encoding="utf-8") as f:
        state = json.load(f)
        
    scenario_id = state["scenario_id"]
    messages = state["messages"]
    steps = state["steps"]
    tools_executed = state["tools_executed"]
    tokens_used = state["tokens_used"]
    start_step = state["step_number"] + 1
    
    start_time = time.monotonic()
    
    for step_num in range(start_step, max_iterations + 1):
        if (time.monotonic() - start_time) >= time_limit_seconds:
            return {"scenario_id": scenario_id, "status": "FALLBACK_TIMEOUT", "tools_executed": tools_executed}
            
        try:
            response = chat(messages, tools=TOOLS_SPEC)
            tokens_used += response.usage.total_tokens
        except Exception as e:
            return {"scenario_id": scenario_id, "status": "FALLBACK_ERROR", "tools_executed": tools_executed}
            
        if tokens_used > token_limit:
            return {"scenario_id": scenario_id, "status": "FALLBACK_TOKEN_LIMIT", "tools_executed": tools_executed}
            
        message = response.choices[0].message
        
        if not message.tool_calls or len(message.tool_calls) == 0:
            final_content = message.content or ""
            status_word = final_content.split(":")[0].strip() if ":" in final_content else "UNKNOWN"
            return {
                "scenario_id": scenario_id,
                "status": status_word,
                "final_answer": final_content,
                "steps": steps,
                "tools_executed": tools_executed
            }
            
        tool_calls_raw = []
        for tc in message.tool_calls:
            args = tc.arguments
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            tool_calls_raw.append({"id": tc.id, "name": tc.name, "arguments": args})
            
        messages.append({"role": "assistant", "content": message.content, "tool_calls": tool_calls_raw})
        
        for tc in tool_calls_raw:
            t_name = tc["name"]
            t_args = tc["arguments"]
            tools_executed.append(t_name)
            
            if t_name in TOOL_MAP:
                try:
                    obs = TOOL_MAP[t_name](**t_args)
                except Exception as ex:
                    obs = {"error": str(ex)}
            else:
                obs = {"error": f"Tool {t_name} not found"}
                
            obs_json = json.dumps(obs)
            steps.append({"step": step_num, "tool": t_name, "arguments": t_args, "observation": obs})
            messages.append({"role": "tool", "tool_call_id": tc["id"], "name": t_name, "content": obs_json})
            
    return {"scenario_id": scenario_id, "status": "FALLBACK_MAX_ITERATIONS", "tools_executed": tools_executed}

def demonstrate_crash_and_resume():
    print("\n--- Running Crash & Resume Demo for S1 ---")
    run_id = "s1_crash_demo"
    eval_path = os.path.join("data", "raw", "agent_scenarios.csv")
    with open(eval_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        row = next(reader)
        user_message = row["message"]
        sc_id = row["scenario_id"]
        
    reset()
    messages = [
        {"role": "system", "content": "AGENT: You are an autonomous refund assistant for Swiggy Instamart."},
        {"role": "user", "content": user_message}
    ]
    
    response = chat(messages, tools=TOOLS_SPEC)
    message = response.choices[0].message
    tc = message.tool_calls[0]
    args = json.loads(tc.arguments) if isinstance(tc.arguments, str) else tc.arguments
    
    obs = TOOL_MAP[tc.name](**args)
    tools_executed = [tc.name]
    steps = [{"step": 1, "tool": tc.name, "arguments": args, "observation": obs}]
    
    messages.append({"role": "assistant", "content": message.content, "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": args}]})
    messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": json.dumps(obs)})
    
    response2 = chat(messages, tools=TOOLS_SPEC)
    message2 = response2.choices[0].message
    tc2 = message2.tool_calls[0]
    args2 = json.loads(tc2.arguments) if isinstance(tc2.arguments, str) else tc2.arguments
    
    obs2 = TOOL_MAP[tc2.name](**args2)
    tools_executed.append(tc2.name)
    steps.append({"step": 2, "tool": tc2.name, "arguments": args2, "observation": obs2})
    
    messages.append({"role": "assistant", "content": message2.content, "tool_calls": [{"id": tc2.id, "name": tc2.name, "arguments": args2}]})
    messages.append({"role": "tool", "tool_call_id": tc2.id, "name": tc2.name, "content": json.dumps(obs2)})
    
    state_data = {
        "run_id": run_id,
        "scenario_id": sc_id,
        "messages": messages,
        "steps": steps,
        "tools_executed": tools_executed,
        "tokens_used": response.usage.total_tokens + response2.usage.total_tokens,
        "step_number": 2
    }
    save_checkpoint(run_id, state_data)
    print("Simulated crash triggered after 2nd tool execution. Checkpoint saved.")
    
    print("Resuming agent from checkpoint...")
    final_res = resume_agent(run_id)
    print(f"Resumed Final Status: {final_res['status']}")
    print(f"Tools Executed Across Crash & Resume: {final_res['tools_executed']}")

if __name__ == "__main__":
    chunks = chunk_documents()
    prepare_retrieval(chunks)
    print("Running Agent Scenarios...")
    
    # Run regular 7 scenarios
    eval_path = os.path.join("data", "raw", "agent_scenarios.csv")
    os.makedirs("outputs", exist_ok=True)
    ledger_path = os.path.join("outputs", "agent_runs.jsonl")
    if os.path.exists(ledger_path):
        os.remove(ledger_path)
        
    with open(eval_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sc_id = row["scenario_id"]
            msg = row["message"]
            res = run_agent(msg, sc_id, max_iterations=6, token_limit=6000, time_limit_seconds=10)
            with open(ledger_path, "a", encoding="utf-8") as lf:
                lf.write(json.dumps(res) + "\n")
            print(f"Scenario {sc_id} -> Status: {res['status']}, Calls: {res['llm_calls']}, Tokens: {res['tokens_used']}")
            
    # Run Crash & Resume Demo
    demonstrate_crash_and_resume()