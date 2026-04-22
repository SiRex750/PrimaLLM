from __future__ import annotations

import os
import re

from dotenv import load_dotenv
from openai import OpenAI
from transformers import pipeline

from shared.triple import KnowledgeTriple


load_dotenv()


USE_LOCAL_SLM = os.getenv("USE_LOCAL_SLM", "false").lower() == "true"
_LOCAL_SLM_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
_LOCAL_GENERATOR = None


def compress_triples(triples: list[KnowledgeTriple], max_items: int = 5) -> str:
    selected = triples[:max_items]
    if not selected:
        return "No facts available to compress."
    return "\n".join(f"- {triple.as_text()}" for triple in selected)


def build_caveman_prompt(triples: list[KnowledgeTriple]) -> str:
    fact_block = compress_triples(triples)
    return (
        "You are Caveman, a local compression layer. "
        "Summarize the following facts into compact, factual prose without inventing new claims.\n"
        f"{fact_block}"
    )


def generate_caveman_prose(triples: list[KnowledgeTriple]) -> str:
    prompt = build_caveman_prompt(triples)
    if USE_LOCAL_SLM:
        return _generate_with_local_slm(prompt)
    return _generate_with_openai(prompt)


def _generate_with_openai(prompt: str) -> str:
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Return only compressed Caveman-style text. "
                    "No markdown. No conversational filler. No determiners."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    content = (response.choices[0].message.content or "").strip()
    return _enforce_caveman_output(content)


def _generate_with_local_slm(prompt: str) -> str:
    generator = _get_local_generator()
    messages = [
        {
            "role": "system",
            "content": "Return only compressed Caveman-style text. No determiners. No filler.",
        },
        {"role": "user", "content": prompt},
    ]
    output = generator(
        messages,
        max_new_tokens=120,
        do_sample=False,
    )
    generated = ""
    if output:
        generated_text = output[0].get("generated_text", "")
        if isinstance(generated_text, list):
            generated = generated_text[-1].get("content", "").strip() if generated_text else ""
        else:
            generated = str(generated_text).strip()
    return _enforce_caveman_output(generated)


def _get_local_generator():
    global _LOCAL_GENERATOR
    if _LOCAL_GENERATOR is None:
        _LOCAL_GENERATOR = pipeline("text-generation", model=_LOCAL_SLM_NAME)
    return _LOCAL_GENERATOR


def _enforce_caveman_output(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"```.*?```", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"^[\-*\d\.\s]+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(
        r"\b(?:a|an|the)\b",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    filler_patterns = [
        r"\b(?:here is|here are|i think|i can|let me|please note|sure)\b",
        r"\b(?:as an ai|in summary|to summarize)\b",
    ]
    for pattern in filler_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{2,}", "\n", cleaned)
    return cleaned.strip()
