"""
generator/response_generator.py
────────────────────────────────
Generates answers using Ollama.
Supports multi-turn conversation history and streaming.

Changes
───────
• OLLAMA_MODEL configurable via environment variable
• Added stream_ollama() — yields tokens as they arrive (SSE)
• Replaced print() with proper logging
• Added ollama_available alias for backward compatibility
"""

import json
import logging
import os
import textwrap
from typing import Generator

import requests

log = logging.getLogger(__name__)

_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_URL = f"{_OLLAMA_HOST}/api/generate"
OLLAMA_CHAT_URL = f"{_OLLAMA_HOST}/api/chat"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")


# ─────────────────────────────────────────────────────────────────────────────
#  Ollama calls
# ─────────────────────────────────────────────────────────────────────────────

def _call_ollama_chat(messages: list[dict]) -> str | None:
    try:
        response = requests.post(
            OLLAMA_CHAT_URL,
            json={"model": OLLAMA_MODEL, "messages": messages, "stream": False},
            timeout=60,
        )
        if response.status_code == 200:
            return response.json()["message"]["content"].strip()
    except (requests.ConnectionError, requests.Timeout):
        pass
    except Exception as exc:
        log.error("Ollama chat error: %s", exc)
    return None


def _call_ollama_simple(prompt: str) -> str | None:
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=30,
        )
        if response.status_code == 200:
            return response.json().get("response", "").strip()
    except (requests.ConnectionError, requests.Timeout):
        pass
    except Exception as exc:
        log.error("Ollama generate error: %s", exc)
    return None


def stream_ollama(messages: list[dict]) -> Generator[str, None, None]:
    """
    Generator that yields Server-Sent Event strings, one token at a time.
    Use with FastAPI StreamingResponse(media_type="text/event-stream").

    Format of each yielded string:
        data: {"token": "hello"}\n\n
    Final event:
        data: [DONE]\n\n
    """
    try:
        with requests.post(
            OLLAMA_CHAT_URL,
            json={"model": OLLAMA_MODEL, "messages": messages, "stream": True},
            stream=True,
            timeout=60,
        ) as response:
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    token = data.get("message", {}).get("content", "")
                    if token:
                        yield f"data: {json.dumps({'token': token})}\n\n"
                    if data.get("done"):
                        yield "data: [DONE]\n\n"
                        return
                except json.JSONDecodeError:
                    continue

    except (requests.ConnectionError, requests.Timeout) as exc:
        log.warning("Streaming connection error: %s", exc)
        yield "data: [DONE]\n\n"
    except Exception as exc:
        log.error("Streaming error: %s", exc)
        yield "data: [DONE]\n\n"


# ─────────────────────────────────────────────────────────────────────────────
#  Prompt builders
# ─────────────────────────────────────────────────────────────────────────────

def _system_prompt(chunks: list[dict]) -> str:
    files = list({c["file"] for c in chunks})[:10]
    sample = ", ".join(f"{c['name']}() in {c['file']}" for c in chunks[:5])
    return textwrap.dedent(f"""
        You are an expert code analysis assistant for a multi-language codebase.
        The codebase contains {len(chunks)} functions across files including: {', '.join(files[:6])}.
        Some example functions: {sample}.
        Answer developer questions concisely (under 150 words).
        Always mention function names and file paths.
        Be specific about what functions call and what calls them.
    """).strip()


def _user_message(query: str, chunks: list[dict]) -> str:
    context = ""
    for i, chunk in enumerate(chunks[:4]):
        calls_str = ", ".join(chunk.get("calls", [])) or "none"
        context += textwrap.dedent(f"""
            Result {i + 1}: {chunk['name']}() in {chunk['file']} (line {chunk.get('line', '?')})
            Calls: {calls_str}
            Code:
            {chunk.get('code', '(not available)')[:400]}
        """)
    return f"Developer question: {query}\n\nRelevant code found:\n{context}"


def build_messages(
    query: str,
    chunks: list[dict],
    history: list[dict] | None = None,
) -> list[dict]:
    """Build the full message list for Ollama (shared by chat and stream)."""
    messages = [{"role": "system", "content": _system_prompt(chunks)}]
    if history:
        messages.extend(history[-6:])
    messages.append({"role": "user", "content": _user_message(query, chunks)})
    return messages


# ─────────────────────────────────────────────────────────────────────────────
#  Fallback template
# ─────────────────────────────────────────────────────────────────────────────

def _template_answer(chunks: list[dict]) -> str:
    if not chunks:
        return "No relevant code found."
    top = chunks[0]
    lines = [
        f"**{top['name']}()** in `{top['file']}` (line {top.get('line', '?')})"]
    if top.get("calls"):
        lines.append(f"Calls: {', '.join(top['calls'])}")
    if len(chunks) > 1:
        others = [f"`{c['name']}` in `{c['file']}`" for c in chunks[1:3]]
        lines.append(f"Also found: {'; '.join(others)}")
    lines.append(
        f"\n_Tip: Make sure Ollama is running: `ollama run {OLLAMA_MODEL}`_")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_answer(
    query:   str,
    chunks:  list[dict],
    history: list[dict] | None = None,
) -> str:
    if not chunks:
        return "No relevant code found for your query."

    messages = build_messages(query, chunks, history)

    result = _call_ollama_chat(messages)
    if result:
        return result

    result = _call_ollama_simple(_user_message(query, chunks))
    if result:
        return result

    return _template_answer(chunks)


def llm_available() -> bool:
    try:
        r = requests.get(_OLLAMA_HOST, timeout=2)
        return r.status_code == 200
    except Exception:
        return False


# Backward-compatible alias used in app.py (Streamlit)
ollama_available = llm_available
