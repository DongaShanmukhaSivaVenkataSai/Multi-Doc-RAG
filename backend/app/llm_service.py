"""Groq LLM service with asynchronous streaming and conversational query rewriting."""

import json
import logging
from typing import Generator, AsyncGenerator, Optional
from groq import Groq, AsyncGroq
from app.config import settings

logger = logging.getLogger(__name__)

_groq_client: Optional[Groq] = None
_async_groq_client: Optional[AsyncGroq] = None

SYSTEM_PROMPT = """You are an expert AI assistant that answers questions based on the provided context documents.

## Instructions:
1. Answer the question using ONLY the information from the provided context chunks.
2. If the context doesn't contain enough information to fully answer, say so clearly.
3. Be precise, thorough, and well-organized in your responses.
4. Use markdown formatting for clarity (headers, bullet points, bold text).
5. If multiple documents provide relevant information, synthesize them coherently.
6. Attribution: When making factual claims, cite the source document and page number in brackets, for example: `[DocumentName, p.1]`.
"""


def get_groq_client() -> Groq:
    """Get or initialize synchronous Groq client (singleton)."""
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=settings.groq_api_key)
        logger.info("Synchronous Groq client initialized")
    return _groq_client


def get_async_groq_client() -> AsyncGroq:
    """Get or initialize asynchronous Groq client (singleton)."""
    global _async_groq_client
    if _async_groq_client is None:
        _async_groq_client = AsyncGroq(api_key=settings.groq_api_key)
        logger.info("Asynchronous Groq client initialized")
    return _async_groq_client


def build_context_prompt(chunks: list[dict]) -> str:
    """Build the context section from retrieved chunks."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        doc_name = chunk.get("doc_name", "Unknown")
        page = chunk.get("page", "N/A")
        score = chunk.get("rerank_score", chunk.get("rrf_score", chunk.get("score", 0)))
        context_parts.append(
            f"### Context Chunk {i} [Source: {doc_name}, Page {page}] "
            f"(Relevance Score: {score:.4f})\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(context_parts)


def get_model_candidates() -> list[str]:
    """Candidate models to try in order of priority."""
    candidates = [settings.llm_model]
    for m in ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b", "llama-3.3-70b-versatile"]:
        if m not in candidates:
            candidates.append(m)
    return candidates


async def condense_query(
    question: str,
    chat_history: list[dict] | None = None,
) -> str:
    """
    Reformulate conversational follow-up questions into standalone search queries.
    E.g. Turn 1: 'Tell me about the AI Engineer role', Turn 2: 'What is the CGPA for it?'
    -> Rephrases to: 'What is the required CGPA for the AI Engineer role?'
    """
    clean_q = question.strip()
    if not chat_history or len(chat_history) < 2:
        return clean_q

    # Check if API key is configured
    if not settings.groq_api_key:
        return clean_q

    client = get_async_groq_client()
    recent_history = chat_history[-4:]
    history_text = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in recent_history])

    prompt = f"""Given the following conversation history and follow-up question, rephrase the follow-up question into a standalone, self-contained search query.
Do NOT answer the question. Only return the standalone rephrased query.

Conversation History:
{history_text}

Follow-up Question: {clean_q}

Standalone Query:"""

    for model_name in get_model_candidates():
        try:
            completion = await client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100,
            )
            rewritten = completion.choices[0].message.content.strip()
            logger.info(f"Query condensed ({model_name}): '{clean_q}' -> '{rewritten}'")
            return rewritten if rewritten else clean_q
        except Exception as e:
            err_msg = str(e).lower()
            if "model_not_found" in err_msg or "does not exist" in err_msg or "404" in err_msg:
                continue
            logger.warning(f"Could not condense query: {e}. Falling back to raw query.")
            return clean_q
    return clean_q


async def generate_streaming_async(
    question: str,
    chunks: list[dict],
    chat_history: list[dict] | None = None,
) -> AsyncGenerator[str, None]:
    """
    Generate an asynchronous streaming response from Groq using retrieved context.
    Yields chunks of the response as they arrive without blocking the event loop.
    Includes automatic fallback across available Groq model tiers.
    """
    client = get_async_groq_client()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if chat_history:
        for msg in chat_history[-6:]:
            messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            })

    context = build_context_prompt(chunks)
    user_message = f"""## Retrieved Context:

{context}

## Question:
{question}

Please provide a comprehensive answer based on the context above."""

    messages.append({"role": "user", "content": user_message})

    stream = None
    last_err = None
    for model_name in get_model_candidates():
        try:
            logger.info(f"Streaming from Groq ({model_name})")
            stream = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.1,
                max_tokens=4096,
                stream=True,
            )
            break
        except Exception as e:
            last_err = e
            err_msg = str(e).lower()
            if "model_not_found" in err_msg or "does not exist" in err_msg or "404" in err_msg:
                logger.warning(f"Model '{model_name}' not available on Groq, trying next candidate...")
                continue
            logger.error(f"Groq API error on '{model_name}': {e}")
            yield f"\n\n❌ Error generating response: {str(e)}"
            return

    if stream is None:
        logger.error(f"All candidate Groq models failed. Last error: {last_err}")
        yield f"\n\n❌ Error generating response: {str(last_err)}"
        return

    try:
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        logger.error(f"Groq streaming error: {e}")
        yield f"\n\n❌ Error streaming response: {str(e)}"


def generate_streaming(
    question: str,
    chunks: list[dict],
    chat_history: list[dict] | None = None,
) -> Generator[str, None, None]:
    """Synchronous generator wrapper with automatic model fallback."""
    client = get_groq_client()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if chat_history:
        for msg in chat_history[-6:]:
            messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})

    context = build_context_prompt(chunks)
    user_message = f"## Retrieved Context:\n\n{context}\n\n## Question:\n{question}\n\nPlease provide a comprehensive answer based on the context above."
    messages.append({"role": "user", "content": user_message})

    stream = None
    last_err = None
    for model_name in get_model_candidates():
        try:
            stream = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.1,
                max_tokens=4096,
                stream=True,
            )
            break
        except Exception as e:
            last_err = e
            err_msg = str(e).lower()
            if "model_not_found" in err_msg or "does not exist" in err_msg or "404" in err_msg:
                continue
            logger.error(f"Groq API error: {e}")
            yield f"\n\n❌ Error generating response: {str(e)}"
            return

    if stream is None:
        yield f"\n\n❌ Error generating response: {str(last_err)}"
        return

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


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
