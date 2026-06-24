import re
import time
from google import genai
from google.genai import types
from loguru import logger

from config import settings
from agent.state import AgentState

_client = genai.Client(api_key=settings.gemini_api_key)

CONVERSATIONAL_INTENTS = {"conversational", "greeting", "farewell", "thanks", "smalltalk"}

# Patterns that are unambiguously conversational — no LLM call needed
_CONVERSATIONAL_RE = re.compile(
    r"^\s*("
    r"hi+|hey+|hello+|howdy|yo|sup|hiya"
    r"|good\s+(morning|afternoon|evening|night|day)"
    r"|how\s+are\s+(you|u)(\s+doing|today)?"
    r"|what'?s\s+up"
    r"|thanks?(\s+you)?|thank\s+you|ty|thx|cheers"
    r"|bye+|goodbye|see\s+ya|later|cya|farewell"
    r"|nice|great|awesome|cool|ok+|okay|got\s+it|sounds?\s+good"
    r"|lol|haha|:?\)"
    r")\s*[!?.]?\s*$",
    re.IGNORECASE,
)

ROUTER_PROMPT = """Classify the user's message intent. Reply with EXACTLY one label from this list, nothing else:

conversational  — greetings, small talk, thanks, farewells, casual chat ("how are you", "nice", "great")
document_qa     — question requiring specific facts from uploaded documents
document_summary — request to summarize a document
comparison      — compare two or more things from documents
multi_hop       — complex multi-step question needing several retrievals
web_search      — query about recent events, news, or external URLs
clarification   — too vague to answer without clarification

User message: "{query}"
Recent history: {history}

Label:"""

CONVERSATIONAL_SYSTEM = (
    "You are a friendly, helpful AI assistant. "
    "For greetings and small talk, respond warmly and briefly. "
    "Never mention documents, files, retrieval, or knowledge bases."
)


async def run(state: AgentState) -> AgentState:
    t0 = time.time()
    query = state["query"].strip()

    # Fast path: keyword match — no LLM needed
    if _CONVERSATIONAL_RE.match(query):
        intent = "conversational"
        logger.info(f"[router] query='{query}' intent='{intent}' (keyword match) t={time.time()-t0:.3f}s")
    else:
        intent = _classify_with_llm(query, state)

    elapsed = time.time() - t0
    logger.info(f"[router] query='{query}' intent='{intent}' t={elapsed:.2f}s")

    state["intent"] = intent
    state["trace"]["router"] = {"intent": intent, "latency_s": elapsed}

    if intent in CONVERSATIONAL_INTENTS:
        await _handle_conversational(state)

    return state


def _classify_with_llm(query: str, state: AgentState) -> str:
    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in state.get("conversation_history", [])[-2:]
    ) or "None"

    try:
        response = _client.models.generate_content(
            model=settings.llm_model,
            contents=ROUTER_PROMPT.format(query=query, history=history_text),
            config=types.GenerateContentConfig(max_output_tokens=16, temperature=0),
        )
        raw = response.text
        if raw is None:
            logger.warning("[router] LLM returned None, defaulting to document_qa")
            return "document_qa"
        label = raw.strip().lower().split()[0]
        # Normalise known labels
        valid = {"conversational", "document_qa", "document_summary", "comparison",
                 "multi_hop", "web_search", "clarification"}
        return label if label in valid else "document_qa"
    except Exception as e:
        logger.warning(f"[router] LLM classification failed: {e}, defaulting to document_qa")
        return "document_qa"


async def _handle_conversational(state: AgentState) -> None:
    query = state["query"]
    q_stream = state.get("stream_queue")

    # Tell frontend this is a conversational turn — hide RAG trace
    if q_stream:
        await q_stream.put({"event": "conversational", "data": {}})

    full_text = ""
    try:
        response = _client.models.generate_content_stream(
            model=settings.llm_model,
            contents=f"Respond naturally and briefly to: {query}",
            config=types.GenerateContentConfig(
                system_instruction=CONVERSATIONAL_SYSTEM,
                max_output_tokens=256,
                temperature=0.85,
            ),
        )
        for chunk in response:
            if chunk.text:
                full_text += chunk.text
                if q_stream:
                    for char in chunk.text:
                        try:
                            q_stream.put_nowait({"event": "token", "data": {"text": char}})
                        except Exception:
                            pass
    except Exception as e:
        logger.error(f"[router] conversational generation failed: {e}")
        full_text = "Hi there! How can I help you?"
        if q_stream:
            for char in full_text:
                try:
                    q_stream.put_nowait({"event": "token", "data": {"text": char}})
                except Exception:
                    pass

    state["answer"] = full_text
    state["citations"] = []
    state["follow_up_questions"] = []
    state["confidence_score"] = 1.0
    state["reflection_passed"] = True
