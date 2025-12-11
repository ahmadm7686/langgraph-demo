# main.py (Ollama-compatible, no errors)
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_community.llms import Ollama
import json
import os

# -----------------
# State schema
# -----------------
class MyState(TypedDict):
    user_input: str
    research_notes: list
    confidence: float
    research_attempts: int
    final_text: str

# -----------------
# JSON memory
# -----------------
MEMORY_FILE = "memory.json"

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def save_memory(state: dict):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

# -----------------
# Ollama LLM
# -----------------
# IMPORTANT: Make sure your Ollama is running locally
# Example model: mistral, llama3, qwen2, deepseek-r1, etc.
llm = Ollama(model="llama3")

# -----------------
# Nodes
# -----------------

def researcher_node(state: MyState):
    attempts = state.get("research_attempts", 0)
    prompt = f"Research about: {state['user_input']} (attempt {attempts+1})
Give short result + numeric confidence (0-1). Format: TEXT || CONFIDENCE:0.x"

    resp = llm.invoke(prompt)

    if "||" in resp:
        text, meta = resp.split("||")
        confidence = float(meta.split(":")[1])
    else:
        text = resp
        confidence = 0.5

    notes = state.get("research_notes", []) + [text.strip()]
    attempts += 1

    print(f"[Researcher] attempt={attempts} confidence={confidence} note='{text.strip()}'")

    return {
        "research_notes": notes,
        "confidence": confidence,
        "research_attempts": attempts
    }


def writer_node(state: MyState):
    conf = state.get("confidence", 0.0)
    notes = state.get("research_notes", [])

    if conf >= 0.7:
        final = f"Summary (conf={conf}): {notes[-1]}"
        print(f"[Writer] produced final text")
        return {"final_text": final}
    else:
        print(f"[Writer] confidence too low ({conf}), not producing final text")
        return {}


def supervisor_node(state: MyState):
    print("[Supervisor] routing to Researcher")
    return {}

# -----------------
# Graph
# -----------------

graph = StateGraph(MyState)

graph.add_node("Supervisor", supervisor_node)
graph.add_node("Researcher", researcher_node)
graph.add_node("Writer", writer_node)

graph.add_edge(START, "Supervisor")
graph.add_edge("Supervisor", "Researcher")


def should_continue(state: MyState) -> bool:
    return state.get("confidence", 0.0) >= 0.7


graph.add_conditional_edge(
    "Researcher",
    should_continue,
    {"true": "Writer", "false": "Researcher"}
)

graph.add_edge("Writer", END)

app = graph.compile()

# -----------------
# Runner
# -----------------
if __name__ == "__main__":
    saved = load_memory()
    if saved:
        print("Loaded memory from disk.")
        state = saved
    else:
        state = {
            "user_input": "Explain how transformers attention works",
            "research_notes": [],
            "confidence": 0.0,
            "research_attempts": 0,
            "final_text": ""
        }

    result = app.invoke(state)

    save_memory(result)

    print("--- Final state ---")
    print(json.dumps(result, indent=2, ensure_ascii=False))
