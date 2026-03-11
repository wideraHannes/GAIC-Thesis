# Learnings

> This file has the purpose to collect the learning that i experience during experimenting

1. Effective Context engineering, Reproducable results in preprocessing
2. Prompt engineering in Zero Shot is Required -> to confident for Argument [[analysis_v1_results]]

# LLMs as Text classifier:

[source](https://wandb.ai/gladiator/LLMs-as-classifiers/reports/LLMs-are-machine-learning-classifiers--VmlldzoxMTEwNzUyNA#why-llms-can-be-effective-for-text-classification)
This seems easy enough, but it comes with a major flaw: decoder-based LLMs are designed to generate freeform text, making their outputs unpredictable. A model might respond with a paragraph-long explanation instead of the expected "positive" or "negative," making integration into structured applications challenging.
Thankfully, there are effective ways to constrain and optimize LLM outputs for classification tasks. In this post, we’ll explore various techniques, including:
Prompt engineering for structured outputs

Few-shot learning to improve reliability

Fine-tuning to align LLM behavior with classification objectives
