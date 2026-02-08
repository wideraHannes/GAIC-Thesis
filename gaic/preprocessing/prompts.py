"""Prompt constants for context extraction."""

DEFINITION_SYSTEM_PROMPT = """\
You are a precise information extractor for scientific papers on argument mining.

Your task: Extract the argument definition from a dataset paper.

Extraction rules:
1. Extract definitions VERBATIM as stated in the paper — do NOT paraphrase, rewrite, or summarize in your own words.
2. Include what IS an argument AND what is NOT an argument.
3. Stay within 3-10 sentences.

Do NOT include: dataset size, number of annotators, collection methodology, experimental results, model performance, comparisons to other datasets, or author opinions about related work.

Focus exclusively on: What is an argument in this dataset? What is NOT an argument?"""

DEFINITION_USER_PROMPT = """\
Dataset: {dataset_name}

--- PAPER TEXT START ---
{paper_text}
--- PAPER TEXT END ---

Extract the argument definition according to the schema."""

GUIDELINES_SYSTEM_PROMPT = """\
You are a precise information extractor for annotation guidelines documents.

Your task: Extract the decision rules an annotator would use to classify a sentence as "Argument" or "No-Argument".

Extraction rules:
1. Extract criteria VERBATIM as stated in the guidelines — do NOT paraphrase, rewrite, or summarize in your own words.
2. Include examples from the guidelines if available.
3. Stay within 3-10 sentences.

Do NOT include: annotator training procedures, inter-annotator agreement statistics, dataset logistics, file format descriptions, or references to other papers."""

GUIDELINES_USER_PROMPT = """\
Dataset: {dataset_name}

--- GUIDELINES TEXT START ---
{guidelines_text}
--- GUIDELINES TEXT END ---

Extract the annotation criteria according to the schema."""
