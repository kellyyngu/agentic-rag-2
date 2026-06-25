import asyncio
import re
import time
from google import genai
from google.genai import types
from loguru import logger

from config import settings
from agent.state import AgentState

_client = genai.Client(api_key=settings.gemini_api_key)

# Intents that bypass the RAG pipeline entirely
DIRECT_INTENTS = {
    "conversational", "greeting", "farewell", "thanks", "smalltalk",
    "assistant_identity", "general_knowledge",
}

# Fast keyword match for unambiguous conversational patterns
_CONVERSATIONAL_RE = re.compile(
    r"^\s*("
    r"hi+|hey+|hello+|howdy|yo|sup|hiya"
    r"|good\s+(morning|afternoon|evening|night|day)"
    r"|how\s+are\s+(you|u)(\s+(?:doing|today))?"
    r"|what'?s\s+up"
    r"|thanks?(\s+you)?|thank\s+you|ty|thx|cheers"
    r"|bye+|goodbye|see\s+ya|later|cya|farewell"
    r"|nice|great|awesome|cool|ok+|okay|got\s+it|sounds?\s+good"
    r"|lol|haha|:?\)"
    r")\s*[!?.]?\s*$",
    re.IGNORECASE,
)


ROUTER_PROMPT = """You are an intent classifier. Respond with EXACTLY ONE label from the list below. Nothing else.

LABELS:
  conversational     — hi, hello, thanks, how are you, small talk, farewells
  assistant_identity — "who are you?", "what can you do?", "are you an AI?"
  general_knowledge  — questions about well-known, widely-documented concepts that appear in textbooks or encyclopedias (e.g. stress, machine learning, photosynthesis, gravity)
  document_qa        — questions about specific projects, proprietary systems, niche technical terms, company names, product names, or acronyms that are NOT common public knowledge and may be in the user's uploaded documents
  document_summary   — "summarize my document", "give me an overview of the file"
  web_search         — needs live/current data: news, stock prices, today's weather, recent events
  clarification      — too vague to answer at all

KEY DISTINCTION — general_knowledge vs document_qa:
- Is this something you'd find in a textbook or Wikipedia? → general_knowledge
- Is this a specific project name, product, acronym, system, or niche term that might be in uploaded docs? → document_qa

EXAMPLES:
  "what is stress"                → general_knowledge   (common concept, in every textbook)
  "what is machine learning"      → general_knowledge   (well-known public concept)
  "explain quantum computing"     → general_knowledge   (well-known public concept)
  "what is OSM PiNN"              → document_qa         (specific project/acronym, likely in docs)
  "what is ACME system"           → document_qa         (specific system name, likely in docs)
  "explain the methodology"       → document_qa         (refers to something in a document)
  "what is in my document"        → document_qa
  "summarize the uploaded PDF"    → document_summary
  "what happened in the news today" → web_search
  "hi there"                      → conversational
  "what is your name"             → assistant_identity
  "do you have a name"            → assistant_identity
  "what model are you"            → assistant_identity
  "are you chatgpt"               → assistant_identity
  "who made you"                  → assistant_identity

User message: "{query}"
Recent history: {history}

Label:"""

SYSTEM_BY_INTENT = {
    "conversational": (
        "You are a friendly, helpful AI assistant. "
        "Respond warmly and briefly to greetings and small talk. "
        "Never mention documents, retrieval, or knowledge bases."
    ),
    "assistant_identity": (
        "You are an Agentic RAG assistant. "
        "You can answer questions from uploaded documents using hybrid retrieval and re-ranking, "
        "answer general knowledge questions directly, and search the web for current information. "
        "Be concise and helpful when explaining your capabilities."
    ),
    "general_knowledge": (
        "You are a knowledgeable AI assistant. "
        "Answer the question clearly and accurately from your training knowledge. "
        "Do not mention documents, retrieval, or uploaded files. "
        "Structure your answer with markdown if it aids clarity."
    ),
}


async def run(state: AgentState) -> AgentState:
    t0 = time.time()
    query = state["query"].strip()

    # Fast path: obvious conversational patterns only
    if _CONVERSATIONAL_RE.match(query):
        intent = "conversational"
        logger.info(f"[router] query='{query}' intent='{intent}' (keyword) t={time.time()-t0:.3f}s")
    else:
        intent = _classify_with_llm(query, state)

    elapsed = time.time() - t0
    logger.info(f"[router] query='{query}' intent='{intent}' t={elapsed:.2f}s")

    state["intent"] = intent
    state["trace"]["router"] = {"intent": intent, "latency_s": elapsed}

    if intent in DIRECT_INTENTS:
        await _handle_direct(state, intent)

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
            config=types.GenerateContentConfig(
                max_output_tokens=64,
                temperature=0,
                # gemini-2.5-flash thinking tokens were consuming the 64-token
                # budget → empty response → None. Disable thinking for classification.
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        raw = response.text
        if raw is None:
            logger.warning("[router] LLM returned None, defaulting to document_qa")
            return "document_qa"
        label = raw.strip().lower().split()[0]
        valid = {
            "conversational", "assistant_identity", "general_knowledge",
            "document_qa", "document_summary", "research", "web_search", "clarification",
        }
        return label if label in valid else "document_qa"
    except Exception as e:
        logger.warning(f"[router] LLM classification failed: {e}, defaulting to document_qa")
        return "document_qa"


async def _handle_direct(state: AgentState, intent: str) -> None:
    """Answer directly from LLM — no retrieval, no citations.

    Uses asyncio.to_thread so the blocking HTTP call doesn't starve the event loop.
    """
    query = state["query"]
    q_stream = state.get("stream_queue")
    system_prompt = SYSTEM_BY_INTENT.get(intent, SYSTEM_BY_INTENT["general_knowledge"])

    # Signal frontend to hide the RAG trace UI
    if q_stream:
        await q_stream.put({"event": "conversational", "data": {"intent": intent}})

    def _generate() -> str:
        response = _client.models.generate_content(
            model=settings.llm_model,
            contents=query,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=8192,
                temperature=0.7,
            ),
        )
        return response.text or ""

    try:
        full_text = await asyncio.to_thread(_generate)
        if q_stream:
            await q_stream.put({"event": "token", "data": {"text": full_text}})
    except Exception as e:
        logger.error(f"[router] direct generation failed: {e}")
        full_text = "I'm sorry, I encountered an error. Please try again."
        if q_stream:
            await q_stream.put({"event": "token", "data": {"text": full_text}})

    state["answer"] = full_text
    state["citations"] = []
    state["follow_up_questions"] = []
    state["confidence_score"] = 1.0
    state["retrieval_confidence"] = 1.0
    state["reflection_passed"] = True
