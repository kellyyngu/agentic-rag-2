import json
import time
from google import genai
from google.genai import types
from loguru import logger

from config import settings
from agent.state import AgentState

_client = genai.Client(api_key=settings.gemini_api_key)

REFLECTOR_PROMPT = """You are a quality evaluator for a RAG system.

Original query: {query}
Generated answer: {answer}
Number of source chunks used: {chunk_count}
Confidence score: {confidence}

Evaluate the answer on:
1. Does it directly address the query?
2. Are claims grounded in the retrieved context?
3. Is there a significant information gap that warrants another retrieval attempt?

Respond ONLY with valid JSON:
{{
  "passed": true,
  "feedback": "Answer is complete and well-grounded.",
  "missing_aspects": [],
  "should_retrieve_more": false
}}"""


async def run(state: AgentState) -> AgentState:
    t0 = time.time()
    iteration = state.get("iteration_count", 0) + 1
    state["iteration_count"] = iteration

    logger.info(f"[reflector] iteration={iteration} confidence={state.get('confidence_score', 0):.2f}")

    if state.get("confidence_score", 0) >= settings.confidence_threshold or iteration >= settings.max_reflection_iterations:
        state["reflection_passed"] = True
        state["reflection_feedback"] = None
        state["trace"]["reflector"] = {
            "passed": True,
            "fast_pass": True,
            "iteration": iteration,
            "latency_s": 0,
        }
        q_stream = state.get("stream_queue")
        if q_stream:
            await q_stream.put({
                "event": "reflection",
                "data": {
                    "passed": True,
                    "confidence": state.get("confidence_score", 0),
                    "iteration": iteration,
                },
            })
        return state

    result = {"passed": True, "feedback": "", "missing_aspects": [], "should_retrieve_more": False}

    try:
        prompt = REFLECTOR_PROMPT.format(
            query=state["query"],
            answer=state.get("answer", "")[:2000],
            chunk_count=len(state.get("retrieved_chunks", [])),
            confidence=state.get("confidence_score", 0),
        )
        response = _client.models.generate_content(
            model=settings.llm_model,
            contents=prompt,
            config=types.GenerateContentConfig(max_output_tokens=512),
        )
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw.strip())
    except Exception as e:
        logger.warning(f"[reflector] LLM failed: {e}, defaulting to passed=True")

    elapsed = time.time() - t0
    passed = result.get("passed", True)
    feedback = result.get("feedback", "")

    if not passed and result.get("missing_aspects"):
        feedback = " ".join(result["missing_aspects"])

    logger.info(f"[reflector] passed={passed} t={elapsed:.2f}s")

    state["reflection_passed"] = passed
    state["reflection_feedback"] = feedback if not passed else None
    state["trace"]["reflector"] = {
        "passed": passed,
        "feedback": feedback,
        "should_retrieve_more": result.get("should_retrieve_more", False),
        "iteration": iteration,
        "latency_s": elapsed,
    }

    q_stream = state.get("stream_queue")
    if q_stream:
        await q_stream.put({
            "event": "reflection",
            "data": {
                "passed": passed,
                "confidence": state.get("confidence_score", 0),
                "iteration": iteration,
            },
        })

    return state
