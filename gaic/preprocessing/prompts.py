"""Prompt constants for context extraction."""

DEFINITION_SYSTEM_PROMPT = """\
You are a precise information synthesizer for scientific papers on argument mining.

Your task: Synthesize a clear, concise definition of what constitutes an argument in this dataset.

Synthesis rules:
1. Distill the definition from the paper into clear, natural prose.
2. Include what IS an argument AND what is NOT an argument.
3. Write 3-10 sentences in a cohesive, flowing paragraph.

Do NOT include: dataset size, number of annotators, collection methodology, experimental results, model performance, comparisons to other datasets, or author opinions about related work.

This is used as prompt context for an LLM to classify sentences as "Argument" or "No-Argument".

Focus exclusively on: What is an argument in this dataset? What is NOT an argument?"""

DEFINITION_USER_PROMPT = """\
Dataset: {dataset_name}

--- PAPER TEXT START ---
{paper_text}
--- PAPER TEXT END ---

Synthesize the argument definition according to the schema."""

GUIDELINES_SYSTEM_PROMPT = """\
You are a precise information synthesizer for annotation guidelines documents.

Your task: Synthesize the decision rules an annotator would use to classify a sentence as "Argument" or "No-Argument".

Synthesis rules:
1. Distill the key criteria into clear, natural prose.
2. Include examples from the guidelines if available.
3. Write 3-10 sentences in a cohesive, flowing paragraph.
4. Use your own words to create a clean summary — avoid copying quoted phrases verbatim.

Do NOT include: annotator training procedures, inter-annotator agreement statistics, dataset logistics, file format descriptions, or references to other papers."""

GUIDELINES_USER_PROMPT = """\
Dataset: {dataset_name}

--- GUIDELINES TEXT START ---
{guidelines_text}
--- GUIDELINES TEXT END ---

Synthesize the annotation criteria according to the schema."""
