from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from evaluation.evaluator import Evaluator

router = APIRouter()
_evaluator = Evaluator()


class EvalItem(BaseModel):
    query: str
    expected_answer: str
    ground_truth_sources: List[str] = []


class EvalRequest(BaseModel):
    items: List[EvalItem]


class EvalResult(BaseModel):
    query: str
    precision_at_5: float
    recall_at_5: float
    groundedness: float
    citation_accuracy: float
    latency_s: float


@router.post("/evaluate", response_model=List[EvalResult])
async def run_evaluation(request: Request, body: EvalRequest):
    if not body.items:
        raise HTTPException(400, "No evaluation items provided")

    retriever = request.app.state.retriever
    results = []

    for item in body.items:
        result = await _evaluator.evaluate_single(
            query=item.query,
            expected_answer=item.expected_answer,
            ground_truth_sources=item.ground_truth_sources,
            retriever=retriever,
        )
        results.append(EvalResult(**result))

    return results


@router.get("/evaluate/dataset")
async def get_sample_dataset():
    """Return a sample evaluation dataset for demo purposes."""
    return _evaluator.get_sample_dataset()
