import os
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from app.services.embedding_service import EmbeddingService

cheap_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
llm = ChatOpenAI(model="gpt-4o-mini")
embedding_service = EmbeddingService()

class ChunkSearchState(TypedDict):
    label_question: str
    user_id: str
    resume_chunks: List[Dict[str, Any]]
    personal_info_chunks: List[Dict[str, Any]]
    current_chunk: Dict[str, Any]
    extracted_value: str
    confidence: int
    source: str
    logs: List[str]
    chunk_index: int
    phase: str  # "resume", "personal_info", "llm"

# Node 1: Vector search resume
def vector_search_resume_node(state: ChunkSearchState) -> ChunkSearchState:
    label = state["label_question"]
    user_id = state["user_id"]
    resume_results = embedding_service.search_similar_by_document_type(
        query=label, user_id=user_id, document_type="resume", top_k=5
    )
    state["resume_chunks"] = resume_results
    state["chunk_index"] = 0
    state["phase"] = "resume"
    state["logs"].append(f"Resume vector search found {len(resume_results)} chunks.")
    return state

# Node 2: Chunk confidence scoring (for both resume and personal info)
def chunk_confidence_node(state: ChunkSearchState) -> ChunkSearchState:
    label = state["label_question"]
    phase = state["phase"]
    chunks = state["resume_chunks"] if phase == "resume" else state["personal_info_chunks"]
    idx = state["chunk_index"]
    if idx >= len(chunks):
        state["logs"].append(f"No more chunks in {phase}.")
        return state
    chunk = chunks[idx]
    chunk_text = chunk.get("text", "")
    prompt = f"""
You are a confidence scoring assistant. Given a question and a chunk of text, return ONLY a number between 0 and 100 (inclusive) representing your confidence (as a percentage) that the chunk directly and correctly answers the question.

- 100 = perfect match, 0 = not an answer at all.
- Do not explain, do not add any text, just the number.

Examples:
Question: What is your email?
Chunk: ethanabduraimov@gmail.com
Confidence: 100

Question: What is your last name?
Chunk: lastname Abduraimov
Confidence: 100

Now score:
Question: {label}
Chunk: {chunk_text}
Confidence:
"""
    response = cheap_llm.invoke(prompt)
    try:
        import re
        match = re.search(r"([0-9]{1,3})", response.content)
        conf_percent = int(match.group(1)) if match else 50
    except Exception:
        conf_percent = 50
    state["logs"].append(f"Chunk {idx} ({phase}) confidence: {conf_percent}% | Text: '{chunk_text[:80]}'")
    if conf_percent == 100:
        # Use LLM to extract the precise answer from the chunk
        extract_prompt = f"""Extract the answer to the following question from the provided text.\nQuestion: {label}\nText: {chunk_text}\n\nIMPORTANT:\n- Return ONLY the answer, with no explanations, no extra text, and no commentary.\n- If the answer is not present, return 'Not found'.\n\nAnswer:"""
        extract_response = llm.invoke(extract_prompt)
        extracted = extract_response.content.strip()
        state["extracted_value"] = extracted
        state["confidence"] = 100
        state["source"] = phase
        state["logs"].append(f"✅ Found direct answer in {phase} chunk {idx}. Extracted: {extracted}")
    else:
        state["chunk_index"] += 1
    state["current_chunk"] = chunk
    return state

# Node 3: Vector search personal info
def vector_search_personal_info_node(state: ChunkSearchState) -> ChunkSearchState:
    label = state["label_question"]
    user_id = state["user_id"]
    personal_results = embedding_service.search_similar_by_document_type(
        query=label, user_id=user_id, document_type="personal_info", top_k=5
    )
    state["personal_info_chunks"] = personal_results
    state["chunk_index"] = 0
    state["phase"] = "personal_info"
    state["logs"].append(f"Personal info vector search found {len(personal_results)} chunks.")
    return state

# Node 4: LLM generative fallback
def llm_generate_node(state: ChunkSearchState) -> ChunkSearchState:
    label = state["label_question"]
    context = "\n".join([
        c.get("text", "") for c in state.get("resume_chunks", [])[:2] + state.get("personal_info_chunks", [])[:2]
    ])
    prompt = f"""Extract the answer to the following question from the provided text.\n\nQuestion: {label}\nText: {context}\n\nIMPORTANT:\n- Return ONLY the answer, with no explanations, no extra text, and no commentary.\n- If the answer is not present, return 'Not found'.\n\nAnswer:"""
    response = llm.invoke(prompt)
    answer = response.content.strip()
    state["extracted_value"] = answer
    state["confidence"] = 0
    state["source"] = "llm"
    state["logs"].append(f"LLM fallback answer: {answer}")
    return state

# Routing logic matching the diagram
def route_after_chunk_confidence(state: ChunkSearchState) -> str:
    phase = state["phase"]
    chunks = state["resume_chunks"] if phase == "resume" else state["personal_info_chunks"]
    idx = state["chunk_index"]
    if state.get("confidence", 0) == 100:
        state["logs"].append("[ROUTE] 100% confidence, returning answer.")
        return "return_answer"
    if idx < len(chunks):
        if phase == "resume":
            state["logs"].append("[ROUTE] <100% confidence, more resume chunks left. Next chunk.")
            return "next_chunk_or_personal_info"
        else:
            state["logs"].append("[ROUTE] <100% confidence, more personal info chunks left. Next chunk.")
            return "next_chunk_or_llm"
    if phase == "resume":
        state["logs"].append("[ROUTE] No resume chunks left. Switch to personal info.")
        return "vector_search_personal_info"
    if phase == "personal_info":
        state["logs"].append("[ROUTE] No personal info chunks left. Switch to LLM.")
        return "llm_generate"
    state["logs"].append("[ROUTE] End of pipeline.")
    return "return_answer"

# Return answer node (for clarity)
def return_answer_node(state: ChunkSearchState) -> ChunkSearchState:
    state["logs"].append("[RETURN] Returning final answer.")
    return state

# Build the graph to match the diagram
builder = StateGraph(ChunkSearchState)
builder.add_node("vector_search_resume", vector_search_resume_node)
builder.add_node("chunk_confidence", chunk_confidence_node)
builder.add_node("vector_search_personal_info", vector_search_personal_info_node)
builder.add_node("llm_generate", llm_generate_node)
builder.add_node("return_answer", return_answer_node)

builder.add_edge(START, "vector_search_resume")
builder.add_edge("vector_search_resume", "chunk_confidence")

# Resume chunk routing
builder.add_conditional_edges(
    "chunk_confidence",
    route_after_chunk_confidence,
    {
        "next_chunk_or_personal_info": "chunk_confidence",  # next resume chunk
        "vector_search_personal_info": "vector_search_personal_info",
        "return_answer": "return_answer",
        "next_chunk_or_llm": "chunk_confidence",  # next personal info chunk
        "llm_generate": "llm_generate"
    }
)

# After personal info search, go to chunk_confidence
builder.add_edge("vector_search_personal_info", "chunk_confidence")
# After LLM fallback, return answer
builder.add_edge("llm_generate", "return_answer")

# Compile the graph
chunk_confidence_graph = builder.compile()

def extract_with_chunk_confidence(label_question: str, user_id: str) -> Dict[str, Any]:
    state: ChunkSearchState = {
        "label_question": label_question,
        "user_id": user_id,
        "resume_chunks": [],
        "personal_info_chunks": [],
        "current_chunk": {},
        "extracted_value": "",
        "confidence": 0,
        "source": "",
        "logs": [],
        "chunk_index": 0,
        "phase": "resume"
    }
    final_state = chunk_confidence_graph.invoke(state)
    return {
        "answer": final_state["extracted_value"],
        "confidence": final_state["confidence"],
        "source": final_state["source"],
        "logs": final_state["logs"]
    } 