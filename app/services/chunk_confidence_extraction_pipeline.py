import os
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from app.services.embedding_service import EmbeddingService
import re

cheap_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
llm = ChatOpenAI(model="gpt-4o-mini")
embedding_service = EmbeddingService()

def regex_extract(label: str, chunk_text: str) -> str:
    # Example: extract email if label contains 'email'
    if 'email' in label.lower():
        match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", chunk_text)
        return match.group(0) if match else chunk_text
    # Add more field-specific regex as needed
    return chunk_text

def clean_extracted_answer(label: str, extracted: str) -> str:
    # Remove any prefix ending with 'name' (e.g., 'lastname:', 'first name:', etc.)
    extracted_clean = extracted.strip()
    # Regex: remove any prefix like 'lastname:', 'first name:', 'name:', etc.
    extracted_clean = re.sub(r'^[a-zA-Z ]*name:?\s*', '', extracted_clean, flags=re.IGNORECASE)
    return extracted_clean.strip(" :,-")

def confidence_scorer(label: str, answer: str) -> int:
    prompt = f"""
You are a confidence scoring assistant. Given a question and an answer, return ONLY a number between 0 and 100 (inclusive) representing your confidence (as a percentage) that the answer directly and correctly answers the question.

- 100 = perfect match, 0 = not an answer at all.
- Do not explain, do not add any text, just the number.

Examples:
Question: What is your email?
Answer: ethanabduraimov@gmail.com
Confidence: 100

Question: What is your last name?
Answer: Abduraimov
Confidence: 100

Now score:
Question: {label}
Answer: {answer}
Confidence:
"""
    response = cheap_llm.invoke(prompt)
    match = re.search(r"([0-9]{1,3})", response.content)
    return int(match.group(1)) if match else 50

def llm_extract(label: str, context: str) -> str:
    prompt = f"""Extract the answer to the following question from the provided text.\n\nQuestion: {label}\nText: {context}\n\nIMPORTANT:\n- Return ONLY the answer, with no explanations, no extra text, and no commentary.\n- If the answer is not present, return 'Not found'.\n\nAnswer:"""
    response = llm.invoke(prompt)
    return response.content.strip()

def scan_raw_doc(label: str, user_id: str) -> str:
    # Fallback: LLM generates a best-effort answer from the label/question only
    prompt = (
        f"You are a professional assistant. Generate a helpful, context-aware answer to the following question "
        f"for a job application form. If the question is open-ended, provide a positive, relevant response. "
        f"Question: {label}\n\n"
        f"IMPORTANT:\n- Do not say 'I am an AI language model'.\n- Be concise and professional.\n\n"
        f"Answer:"
    )
    response = llm.invoke(prompt)
    return response.content.strip()

def extract_with_chunk_confidence(label_question: str, user_id: str) -> Dict[str, Any]:
    logs = []
    # 1. Gather all chunks (resume + personal info)
    resume_chunks = embedding_service.search_similar_by_document_type(
        query=label_question, user_id=user_id, document_type="resume", top_k=10
    )
    personal_chunks = embedding_service.search_similar_by_document_type(
        query=label_question, user_id=user_id, document_type="personal_info", top_k=10
    )
    all_chunks = resume_chunks + personal_chunks
    logs.append(f"Found {len(all_chunks)} total chunks (resume + personal info)")

    # 2. Loop over all chunks
    for idx, chunk in enumerate(all_chunks):
        chunk_text = chunk.get("text", "")
        logs.append(f"[Chunk {idx}] Text: {chunk_text[:80]}")
        extracted = regex_extract(label_question, chunk_text)
        cleaned = clean_extracted_answer(label_question, extracted)
        conf = confidence_scorer(label_question, cleaned)
        logs.append(f"[Chunk {idx}] Extracted: {cleaned} | Confidence: {conf}%")
        if conf >= 90:
            logs.append(f"✅ High confidence answer found in chunk {idx}: {cleaned}")
            return {
                "answer": cleaned,
                "confidence": conf,
                "source": f"vector_chunk_{idx}",
                "logs": logs
            }

    logs.append("❌ No high-confidence answer found in any chunk. Trying LLM extract.")
    # 3. LLM extract fallback
    context = "\n".join([c.get("text", "") for c in all_chunks[:4]])
    llm_answer = llm_extract(label_question, context)
    llm_cleaned = clean_extracted_answer(label_question, llm_answer)
    llm_conf = confidence_scorer(label_question, llm_cleaned)
    logs.append(f"LLM extracted: {llm_cleaned} | Confidence: {llm_conf}%")
    if llm_conf >= 90:
        logs.append(f"✅ High confidence LLM answer: {llm_cleaned}")
        return {
            "answer": llm_cleaned,
            "confidence": llm_conf,
            "source": "llm_extract",
            "logs": logs
        }

    logs.append("❌ LLM answer not high confidence. Trying raw doc scan.")
    # 4. Raw doc scan fallback
    fallback_answer = scan_raw_doc(label_question, user_id)
    fallback_cleaned = clean_extracted_answer(label_question, fallback_answer)
    fallback_conf = confidence_scorer(label_question, fallback_cleaned)
    logs.append(f"Raw doc scan extracted: {fallback_cleaned} | Confidence: {fallback_conf}%")
    logs.append("Returning best effort answer.")
    return {
        "answer": fallback_cleaned,
        "confidence": fallback_conf,
        "source": "raw_doc_scan",
        "logs": logs
    } 