"""Pydantic schemas for structured LLM extraction output."""

from pydantic import BaseModel, Field


class DefinitionExtraction(BaseModel):
    definition: str = Field(
        description=(
            "3-10 sentences defining what an argument is and what it is not "
            "in this dataset. Synthesize a clear, cohesive summary in natural prose."
        )
    )


class GuidelinesExtraction(BaseModel):
    guidelines: str = Field(
        description=(
            "3-10 sentences summarizing the instructions given to annotators "
            "for deciding whether a sentence is an argument or not. "
            "Include examples from the guidelines if available. "
            "Synthesize a clear, cohesive summary in natural prose."
        )
    )
