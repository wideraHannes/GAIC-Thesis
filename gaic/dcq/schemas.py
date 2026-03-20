"""Pydantic schemas for DCQ (Data Contamination Quiz) experiment."""

from pydantic import BaseModel, Field
from typing import Literal


class PerturbationSet(BaseModel):
    """Four synonym-based perturbations of an original sentence."""

    perturbation_1: str = Field(..., description="First perturbation")
    perturbation_2: str = Field(..., description="Second perturbation")
    perturbation_3: str = Field(..., description="Third perturbation")
    perturbation_4: str = Field(..., description="Fourth perturbation")


class QuizAnswer(BaseModel):
    """Answer to a multiple-choice quiz question."""

    answer: Literal["A", "B", "C", "D", "E"] = Field(
        ..., description="Selected option letter"
    )
