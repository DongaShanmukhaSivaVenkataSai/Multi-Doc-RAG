"""Groq LLM service with streaming support."""

import json
import logging
from typing import Generator
from groq import Groq
from app.config import settings

logger = logging.getLogger(__name__)

_groq_client = None

SYSTEM_PROMPT = """You are an expert AI assistant that answers questions based on the provided context documents. 

## Instructions:
1. Answer the question using ONLY the information from the provided context chunks.
2. If the context doesn't contain enough information to fully answer, say so clearly.
3. Be precise, thorough, and well-organized in your responses.
4. Use markdown formatting for clarity (headers, bullet points, bold text).
5. If multiple documents provide relevant information, synthesize them coherently.
6. CRITICAL: Do NOT include any source citations, document names, page numbers, or references (like "[Source: ...]" or "(Document: ...)") inside your generated text. The UI handles citations automatically. Just provide the direct answer.
"""


def get_groq_client() -> Groq:
    """Get or initialize the Groq client (singleton)."""
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=settings.groq_api_key)
        logger.info("Groq client initialized")
    return _groq_client


def build_context_prompt(chunks: list[dict]) -> str:
    """Build the context section from retrieved chunks."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        doc_name = chunk.get("doc_name", "Unknown")
        page = chunk.get("page", "N/A")
        score = chunk.get("rerank_score", chunk.get("rrf_score", chunk.get("score", 0)))
        context_parts.append(
            f"### Context Chunk {i} [Source: {doc_name}, Page {page}] "
            f"(Relevance: {score:.4f})\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(context_parts)


def generate_streaming(
    question: str,
    chunks: list[dict],
    chat_history: list[dict] | None = None,
) -> Generator[str, None, None]:
    """
    Generate a streaming response from Groq using retrieved context.

    Yields chunks of the response as they arrive from the API.
    """
    client = get_groq_client()

    # Build messages
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add chat history for conversational context
    if chat_history:
        for msg in chat_history[-6:]:  # Keep last 3 turns (6 messages)
            messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            })

    # Build the final user message with context
    context = build_context_prompt(chunks)
    user_message = f"""## Retrieved Context:

{context}

## Question:
{question}

Please provide a comprehensive answer based on the context above."""

    messages.append({"role": "user", "content": user_message})

    logger.info(f"Sending to Groq ({settings.llm_model}): {len(messages)} messages")

    try:
        stream = client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            temperature=0.1,
            max_tokens=4096,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    except Exception as e:
        logger.error(f"Groq API error: {e}")
        yield f"\n\n❌ Error generating response: {str(e)}"


def generate_response(
    question: str,
    chunks: list[dict],
    chat_history: list[dict] | None = None,
) -> str:
    """Generate a complete (non-streaming) response."""
    full_response = ""
    for token in generate_streaming(question, chunks, chat_history):
        full_response += token
    return full_response


def check_connection() -> bool:
    """Check if Groq API is accessible."""
    try:
        client = get_groq_client()
        client.models.list()
        return True
    except Exception:
        return False
