# V2 Experiment Configuration

Changes from V1:

## Prompt Improvements
- **Context-first placement**: Context now appears at the top of the prompt (better attention)
- **Expert role**: "You are an expert in argumentation analysis"
- **Explicit context ordering**: definition → guideline → document_context
- **Guideline precedence**: "When the definition and guideline conflict, follow the guideline"
- **c0 fallback**: Proper generic criteria when no definition provided

## Structured Output
- **Pydantic model**: Forces valid "Argument" / "No-Argument" labels (no empty responses)
- **Reasoning disabled**: Tested reasoning vs no-reasoning on GPT-4.1 (c1, all 10 datasets) - no significant difference, so disabled for efficiency

## Context Updates
- **AFS definition**: Updated for clarity
- **AEC definition**: Updated for clarity

## Manipulation Fixes
- **Shuffle punctuation**: Period preserved at end of shuffled sentences

## Models
- mistral_medium (in progress)
- (more to be added)
