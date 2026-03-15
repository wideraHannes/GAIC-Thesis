# c0: No Context (Baseline)

## System Prompt

```
You are an expert in argumentation analysis.

## Classification Criteria
Use your general understanding of argumentation.

- "Argument": A sentence that contains a claim supported by reasoning or evidence, expresses a stance, or draws a conclusion.
- "No-Argument": A sentence that is purely factual, procedural, or background information without taking a position.

## Task
Classify whether the following sentence is an argument based on the criteria above.
```

## User Prompt

```
Long-term relief of symptoms has the potential to improve overall quality of life with better compliance for cancer patients.
```

## Expected Output (Structured)

```json
{"label": "Argument"}
```
