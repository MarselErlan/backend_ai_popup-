import os
import json
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from app.services.embedding_service import EmbeddingService
import re

# Enable LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini")
embedding_service = EmbeddingService()
cheap_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)

class ExtractionState(TypedDict):
    label_question: str
    user_id: str
    top_chunks: List[str]
    extracted_value: str
    confidence: float
    reasoning: str
    logs: List[str]

# Node 1: Vector search (resume + personal info)
def vector_search_node(state: ExtractionState) -> ExtractionState:
    label = state["label_question"]
    user_id = state["user_id"]
    # Search both resume and personal info vectors
    resume_results = embedding_service.search_similar_by_document_type(
        query=label, user_id=user_id, document_type="resume", top_k=2
    )
    personal_results = embedding_service.search_similar_by_document_type(
        query=label, user_id=user_id, document_type="personal_info", top_k=2
    )
    # Combine and keep only the text
    top_chunks = [r.get("text", "") for r in resume_results + personal_results if r.get("text")]
    state["top_chunks"] = top_chunks
    # Log the exact chunks (chunk_id and preview)
    chunk_logs = []
    for r in resume_results + personal_results:
        chunk_id = r.get("chunk_id", "?")
        text_preview = r.get("text", "")[:100].replace("\n", " ")
        chunk_logs.append(f"Chunk {chunk_id}: '{text_preview}'")
    state["logs"].append(f"Vector search found {len(top_chunks)} chunks:")
    state["logs"].extend(chunk_logs)
    return state

# Node 2: Dynamic LLM extraction
def dynamic_llm_extract_node(state: ExtractionState) -> ExtractionState:
    label = state["label_question"]
    top_chunks = state["top_chunks"]
    if not top_chunks:
        state["extracted_value"] = "Not found"
        state["confidence"] = 0.0
        state["reasoning"] = "No relevant chunks found."
        state["logs"].append("No chunks to extract from.")
        return state
    # Use the top chunk (or concatenate top 2 for more context)
    context = "\n".join(top_chunks[:2])
    prompt = f"""Extract the answer to the following question from the provided text.\n\nQuestion: {label}\nText: {context}\n\nIMPORTANT:\n- Return ONLY the answer, with no explanations, no extra text, and no commentary.\n- If the answer is not present, return 'Not found'.\n\nAnswer:"""
    response = llm.invoke(prompt)
    answer = response.content.strip()
    state["extracted_value"] = answer
    state["confidence"] = 0.9 if answer and answer.lower() != "not found" else 0.3
    state["reasoning"] = f"LLM extracted answer for '{label}' from top chunk(s)."
    state["logs"].append(f"LLM extraction answer: {answer}")
    return state

# Node 3: Confidence scorer (optional, can be extended)
def confidence_scorer_node(state: ExtractionState) -> ExtractionState:
    label = state["label_question"]
    answer = state["extracted_value"]
    if not answer or answer.lower() == "not found":
        state["confidence"] = 0.0
        state["reasoning"] += " | Low confidence: No answer extracted."
        state["logs"].append("Confidence scored: 0.0 (no answer)")
        return state

    prompt = f"""
You are a confidence scoring assistant. Given a question and an extracted answer, return ONLY a number between 0 and 100 (inclusive) representing your confidence (as a percentage) that the answer directly and correctly answers the question.

- 100 = perfect match, 0 = not an answer at all.
- Do not explain, do not add any text, just the number.

Examples:
Question: What is your email?
Answer: ethanabduraimov@gmail.com
Confidence: 100

Question: What is your email?
Answer: Not found
Confidence: 0

Question: What is your last name?
Answer: Abduraimov
Confidence: 100

Now score:
Question: {label}
Answer: {answer}
Confidence:
"""
    response = cheap_llm.invoke(prompt)
    try:
        match = re.search(r"([0-9]{1,3})", response.content)
        conf_percent = int(match.group(1)) if match else 50
        conf = max(0, min(conf_percent, 100)) / 100.0
        state["confidence"] = conf
        state["logs"].append(f"Confidence scored by LLM: {conf_percent}% ({conf})")
    except Exception:
        state["confidence"] = 0.5
        state["logs"].append("Confidence scoring failed, defaulted to 0.5")
    return state

# Build the graph
builder = StateGraph(ExtractionState)
builder.add_node("vector_search", vector_search_node)
builder.add_node("dynamic_llm_extract", dynamic_llm_extract_node)
builder.add_node("confidence_scorer", confidence_scorer_node)
builder.add_edge(START, "vector_search")
builder.add_edge("vector_search", "dynamic_llm_extract")
builder.add_edge("dynamic_llm_extract", "confidence_scorer")
builder.add_edge("confidence_scorer", END)

# Compile the graph
extraction_graph = builder.compile()

# Example usage function
def extract_answer(label_question: str, user_id: str) -> Dict[str, Any]:
    state: ExtractionState = {
        "label_question": label_question,
        "user_id": user_id,
        "top_chunks": [],
        "extracted_value": "",
        "confidence": 0.0,
        "reasoning": "",
        "logs": []
    }
    final_state = extraction_graph.invoke(state)
    # Convert confidence to percentage for output and logs
    confidence_percent = int(round(final_state["confidence"] * 100))
    logs_with_percent = [
        log.replace(f"{final_state['confidence']}", f"{confidence_percent}%") if "confidence" in log.lower() else log
        for log in final_state["logs"]
    ]
    return {
        "answer": final_state["extracted_value"],
        "confidence": confidence_percent,  # Return as percentage
        "reasoning": final_state["reasoning"],
        "logs": logs_with_percent
    }

# This module can be imported and used in FastAPI endpoints or for testing. 