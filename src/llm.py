"""
LLM generation: builds the citation-forcing prompt and dispatches to
whichever backend is configured/available. Three backends are supported so
the project runs end-to-end even with no API key:

  - "anthropic": Claude via the Anthropic API (needs ANTHROPIC_API_KEY)
  - "openai":    GPT via the OpenAI API (needs OPENAI_API_KEY)
  - "gemini":    Gemini via the Google GenAI API (needs GEMINI_API_KEY)
  - "ollama":    A model running locally via Ollama (https://ollama.com) --
                 free, no API key, no account, runs entirely offline once
                 the model is pulled. Needs the Ollama app running locally.
  - "extractive": no LLM call at all -- returns the highest-scoring
                  retrieved sentences verbatim with citations. This keeps
                  the retrieval half of the project fully testable/demoable
                  without any external dependency or cost.
"""

import os
from dataclasses import dataclass
from typing import List

import config
from src.retriever import RetrievedChunk

SYSTEM_PROMPT = """You are a document Q&A assistant. Answer the user's question using ONLY the \
numbered source excerpts provided below. Follow these rules strictly:

1. Every factual sentence in your answer MUST end with one or more citation tags \
referencing the source(s) it came from, like this: "The policy renews annually [S1]." \
or "Revenue grew 12% [S2][S3]."
2. Do not cite a source for a sentence unless that source actually states the fact.
3. If the sources do not contain enough information to answer the question, say so \
explicitly instead of guessing -- do not use outside knowledge.
4. Be concise and directly answer the question first; don't pad with restated context.
"""


def _build_user_prompt(question: str, retrieved: List[RetrievedChunk]) -> str:
    sources_block = "\n\n".join(
        f"[S{i + 1}] (from {rc.chunk.source}, page {rc.chunk.page_number}):\n{rc.chunk.text}"
        for i, rc in enumerate(retrieved)
    )
    return (
        f"SOURCES:\n{sources_block}\n\n"
        f"QUESTION: {question}\n\n"
        "Answer the question using only the sources above, citing as instructed."
    )


@dataclass
class GeneratedAnswer:
    text: str
    backend_used: str


def _resolve_backend() -> str:
    backend = config.LLM_BACKEND
    if backend != "auto":
        return backend
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini"
    if _ollama_is_running():
        return "ollama"
    return "extractive"


def _ollama_is_running() -> bool:
    """Cheap, fast check so 'auto' mode doesn't hang trying Ollama when it
    isn't installed/running -- a short-timeout GET to its local API."""
    try:
        import urllib.request
        urllib.request.urlopen(f"{_OLLAMA_HOST}/api/tags", timeout=0.3)
        return True
    except Exception:
        return False


_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def _generate_anthropic(question: str, retrieved: List[RetrievedChunk]) -> str:
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=config.MAX_ANSWER_TOKENS,
        temperature=config.GENERATION_TEMPERATURE,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_user_prompt(question, retrieved)}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def _generate_openai(question: str, retrieved: List[RetrievedChunk]) -> str:
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        max_tokens=config.MAX_ANSWER_TOKENS,
        temperature=config.GENERATION_TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(question, retrieved)},
        ],
    )
    return response.choices[0].message.content


def _generate_gemini(question: str, retrieved: List[RetrievedChunk]) -> str:
    """
    Two auth paths are supported because Google's "AQ."-prefixed auth keys
    (the replacement for the old AIzaSy format) have a widely-reported,
    still-open bug as of late 2026 where the Developer API rejects them
    with 401 ACCESS_TOKEN_TYPE_UNSUPPORTED even when used exactly as
    documented (see https://discuss.ai.google.dev -- many threads with
    this identical symptom and no fix yet from Google). If you're stuck
    on an AQ.-only account, set GOOGLE_GENAI_USE_VERTEXAI=true and the
    client will use Vertex AI + Application Default Credentials instead,
    which doesn't depend on the broken key path at all.
    """
    from google import genai
    from google.genai import types

    use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true")

    if use_vertex:
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        if not project:
            raise RuntimeError(
                "GOOGLE_GENAI_USE_VERTEXAI is set but GOOGLE_CLOUD_PROJECT is not. "
                "Vertex AI mode needs a GCP project with the Vertex AI API enabled, "
                "and you must be logged in via `gcloud auth application-default login`."
            )
        client = genai.Client(vertexai=True, project=project, location=location)
    else:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set in the environment. "
                "Run `echo $GEMINI_API_KEY` (or `echo %GEMINI_API_KEY%` on "
                "Windows) to confirm it's actually exported in this shell."
            )
        # vertexai=False is forced explicitly (not left to default) because
        # the SDK also checks GOOGLE_GENAI_USE_VERTEXAI internally -- if that
        # var is set anywhere on the machine, Client() silently switches to
        # Vertex/OAuth mode and ignores api_key, producing the same confusing
        # 401 this whole function is written to avoid.
        client = genai.Client(api_key=api_key, vertexai=False)

    response = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=_build_user_prompt(question, retrieved),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=config.GENERATION_TEMPERATURE,
            max_output_tokens=config.MAX_ANSWER_TOKENS,
        ),
    )
    return response.text


def _generate_ollama(question: str, retrieved: List[RetrievedChunk]) -> str:
    """Call a locally-running Ollama server. No API key, no cost, no
    account -- just `ollama pull <model>` once and `ollama serve` running.
    Uses stdlib urllib instead of requests/an SDK so this backend has zero
    extra pip dependencies."""
    import json
    import urllib.request

    payload = {
        "model": config.OLLAMA_MODEL,
        "prompt": _build_user_prompt(question, retrieved),
        "system": SYSTEM_PROMPT,
        "stream": False,
        "options": {"temperature": config.GENERATION_TEMPERATURE},
    }
    req = urllib.request.Request(
        f"{_OLLAMA_HOST}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        raise RuntimeError(
            f"Could not reach Ollama at {_OLLAMA_HOST}. Is it installed and "
            f"running (`ollama serve`), and is '{config.OLLAMA_MODEL}' pulled "
            f"(`ollama pull {config.OLLAMA_MODEL}`)? Underlying error: {e}"
        )
    return body.get("response", "").strip()


def _generate_extractive(question: str, retrieved: List[RetrievedChunk]) -> str:
    """No-LLM fallback: return the retrieved chunks' most relevant
    sentences verbatim, each tagged with its citation. Deterministic and
    free -- useful for testing/demoing the retrieval pipeline in isolation."""
    if not retrieved:
        return "No relevant information was found in the uploaded documents."

    lines = [
        "(No LLM backend configured -- showing the most relevant excerpts directly.)",
        "",
    ]
    for i, rc in enumerate(retrieved, start=1):
        snippet = rc.chunk.text[:400].strip()
        if len(rc.chunk.text) > 400:
            snippet += "..."
        lines.append(f"{snippet} [S{i}]")
    return "\n\n".join(lines)


_BACKENDS = {
    "anthropic": _generate_anthropic,
    "openai": _generate_openai,
    "gemini": _generate_gemini,
    "ollama": _generate_ollama,
    "extractive": _generate_extractive,
}


def generate_answer(question: str, retrieved: List[RetrievedChunk]) -> GeneratedAnswer:
    if not retrieved:
        return GeneratedAnswer(
            text="I couldn't find anything relevant to that question in the uploaded documents.",
            backend_used="none",
        )

    backend = _resolve_backend()
    try:
        text = _BACKENDS[backend](question, retrieved)
    except Exception as e:
        # Never let an API hiccup take down the whole app -- degrade to
        # the extractive fallback and surface the error in the answer.
        text = (
            f"(LLM backend '{backend}' failed: {e}. Falling back to extractive mode.)\n\n"
            + _generate_extractive(question, retrieved)
        )
        backend = "extractive-fallback"

    return GeneratedAnswer(text=text, backend_used=backend)
