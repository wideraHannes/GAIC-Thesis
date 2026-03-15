# c2: Definition + Guideline

## System Prompt

```
You are an expert in argumentation analysis.

## Argument Definition
In this dataset, an argument consists of explicitly annotated argumentative components in Randomized Controlled Trial abstracts, namely claims (including major claims) and evidence. A claim is defined as a concluding statement made by the author about the outcome of the study, typically describing the relationship between an intervention and a control and derived from the reported results; major claims are more general concluding claims supported by more specific claims and are merged with claims for analysis. Evidence is defined as an observation or measurement in the study, such as outcomes or side effects, which supports or attacks another argumentative component and is considered an observed fact. Argumentative components can be connected by directed relations of support or attack, where support justifies a target component and attack contradicts it or undermines its statistical significance. The paper does not explicitly define any category of non-argumentative text beyond the absence of these annotated components.

## Annotation Guideline
Annotators label a sentence as argumentative if it makes a general, evaluative, or concluding statement about a study outcome or about properties of a treatment/disease that goes beyond reporting raw observations. Concluding comparisons or judgments (e.g., effectiveness, safety, non-inferiority, similarity, trends) are arguments, especially when phrased with cues like "according to the results," "these results support," "this suggests," or contrastive markers like "however" or "but." Pure reporting of measurements, statistics, side effects, or outcomes—even comparative and statistically framed ones—is non-argumentative and treated as premise-like factual reporting, not an argument. Numerical comparisons or significance statements are not arguments unless they are abstracted into a general claim. Overly vague summaries such as "fewer side effects occurred" are not argumentative if underspecified. Multiple properties in one sentence count as multiple arguments only if syntactically separable; inseparable conjunctions are treated as one argument. Boundary examples: (Argument) "According to the results of this study, treatment A was more effective than treatment B"—general conclusion; (Non-argument) "Mean IOP was reduced by 6.5 mm Hg in group A and 6.1 mm Hg in group B"—factual outcome reporting; (Non-argument) "Fewer side effects occurred"—too unspecific to assert a claim.

When the definition and guideline conflict, follow the guideline.

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
