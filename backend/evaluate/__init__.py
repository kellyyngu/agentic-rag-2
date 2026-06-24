"""
Evaluation module for the Agentic RAG system.

This is a pure evaluation LAYER — it does not modify any core RAG logic.
It drives the existing LangGraph orchestrator and measures both:
  * Answer quality   (RAGAS metrics)
  * Decision quality  (custom agentic metrics from execution traces)

Run everything with:   python -m evaluate.run_all
"""
