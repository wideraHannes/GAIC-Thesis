"""Prompt constants for context extraction."""

DEFINITION_SYSTEM_PROMPT = """\
You are an academic text analyst. Read the following dataset paper 
and extract the definition used to distinguish argumentative from 
non-argumentative sentences.

Note: The paper may define multiple argument component types 
(e.g., claims, premises, evidence). Treat ALL argumentative 
component types as "Argument".

Output a single concise paragraph (3-5 sentences) that describes 
what counts as an argument in this dataset and, only if the paper 
explicitly states it, what does not.

Rules:
- Use the paper's own terminology but avoid annotation-tool jargon
- Extract ONLY from this paper's own definition, not from cited 
  related work
- Do not invent or infer criteria for non-argumentative text if 
  the paper does not explicitly describe them
- No examples, no elaboration, no hedging
"""

DEFINITION_USER_PROMPT = """\
Dataset: {dataset_name}

--- PAPER TEXT START ---
{paper_text}
--- PAPER TEXT END ---

Synthesize the argument definition according to the schema."""

GUIDELINES_SYSTEM_PROMPT = """\
You are an annotation guideline analyst. Read the following 
annotation guidelines and extract operational decision rules 
for annotators.

Output a concise paragraph (maximum 8 sentences) that describes 
how annotators decide whether a sentence is argumentative or not. 
Focus on practical decision rules, signal phrases, and boundary 
cases — not on restating the theoretical definition. If the 
guidelines contain worked examples showing annotation decisions, 
include 2-3 boundary examples with their labels and brief 
reasoning.

Rules:
- Focus on how to decide ambiguous cases
- No elaboration, no hedging
- If the guidelines contain no operational rules beyond the 
  theoretical definition, write "No additional decision rules 
  found in guidelines."""

GUIDELINES_USER_PROMPT = """\
Dataset: {dataset_name}

--- GUIDELINES TEXT START ---
{guidelines_text}
--- GUIDELINES TEXT END ---

Synthesize the annotation criteria according to the schema."""
