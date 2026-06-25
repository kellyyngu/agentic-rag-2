"""
Evaluation module for the Agentic RAG system.

This is a pure evaluation LAYER — it does not modify any core RAG logic.
It drives the existing LangGraph orchestrator and measures:
  * Answer quality    (deterministic keyword recall + citation accuracy)
  * Decision quality  (custom agentic metrics from execution traces)
  * Retrieval strategy (BM25 vs vector vs hybrid ablation)

Run everything with:   python -m evaluate.run_all
"""
