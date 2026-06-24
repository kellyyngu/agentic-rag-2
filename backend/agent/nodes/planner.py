import json
import time
from typing import Any
from google import genai
from google.genai import types
from loguru import logger

from config import settings
from agent.state import AgentState

_client = genai.Client(api_key=settings.gemini_api_key)

PLANNER_PROMPT = """You are a query planner for a RAG (Retrieval-Augmented Generation) system.

Analyze the user's query and produce a retrieval plan.

User query: {query}

Conversation history (last 3 turns):
{history}

Your task:
1. Decompose the query into 1–3 specific sub-questions that together answer the full query
2. Identify the retrieval strategy (factual_lookup / comparative / exploratory / conversational)
3. Decide if web search is likely needed (only if the query references recent events or external URLs)

Respond ONLY with valid JSON:
{{
  "sub_questions": ["question 1", "question 2"],
  "retrieval_strategy": "factual_lookup",
  "needs_web_search": false,
  "reasoning": "brief explanation"
}}"""


async def run(state: AgentState) -> AgentState:
    t0 = time.time()
    logger.info(f"[planner] query='{state['query']}'")

    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in state.get("conversation_history", [])[-3:]
    ) or "None"

    prompt = PLANNER_PROMPT.format(
        query=state["query"],
        history=history_text,
    )

    result = {"sub_questions": [state["query"]], "retrieval_strategy": "factual_lookup", "needs_web_search": False}

    try:
        response = _client.models.generate_content(
            model=settings.llm_model,
            contents=prompt,
            config=types.GenerateContentConfig(max_output_tokens=1024),
        )
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw.strip())
    except Exception as e:
        logger.warning(f"[planner] LLM parse failed: {e}, using defaults")

    elapsed = time.time() - t0
    logger.info(f"[planner] sub_questions={result['sub_questions']} strategy={result.get('retrieval_strategy')} t={elapsed:.2f}s")

    state["sub_questions"] = result.get("sub_questions", [state["query"]])
    state["retrieval_strategy"] = result.get("retrieval_strategy", "factual_lookup")
    state["needs_web_search"] = result.get("needs_web_search", False)

    state["trace"]["planner"] = {
        "sub_questions": state["sub_questions"],
        "retrieval_strategy": state["retrieval_strategy"],
        "needs_web_search": state["needs_web_search"],
        "reasoning": result.get("reasoning", ""),
        "latency_s": elapsed,
    }

    q = state.get("stream_queue")
    if q:
        await q.put({
            "event": "plan",
            "data": {
                "sub_questions": state["sub_questions"],
                "strategy": state["retrieval_strategy"],
            },
        })

    return state
