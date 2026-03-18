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

# Huggingface Trainer

For our purpose we use the SFT SuperviseFinetuneTrainer

# Framing of Thesis

research contributes to a broader understanding of: how do we build NLP systems that learn tasks rather than datasets? This connects directly to the foundational concerns about robustness, OOD generalization, and shortcut learning that the entire field is wrestling with.

> For the first time, you can give a model a definition it has never seen during training and ask it to apply that definition. This is closer to how humans work — a human annotator reads the guidelines, understands them, and applies them to new text. If decoders can replicate this for argument mining, that's not just a benchmark improvement. It means:

You could deploy one system across all the application domains I mentioned above, just by swapping the definition in the prompt
New argument mining datasets wouldn't require retraining — just write new guidelines and point the system at them
Cross-lingual transfer becomes more feasible — translate the definition, keep the model
The field moves from building dataset-specific classifiers to building definition-following reasoning systems

# Finetuning

What finetuning Even Makes Sense?

https://meta-pytorch.org/torchtune/stable/tutorials/lora_finetune.html#lora-finetune-label

https://docs.mistral.ai/cookbooks/mistral-fine_tune-mistral_finetune_api

# interesting claim we could do

For Generalization one could claim Worst Performance would be delivering no additional Context, mid performance with argument definition and aannotation guideline and SOTA Performance

# framing

Die korrekte Storyline für die Thesis:

RQ1: "Decoder LLMs zeigen inhärente strukturelle Sensitivität aus dem Pre-Training (Δ_content ≈ 0.10, Δ_order ≈ 0.20) — im Gegensatz zu trainierten Encodern die diese Sensitivität nicht aufweisen (Feger: Δ ≤ 0.02)."
RQ2: "Kontext-Informationen (Definitionen, Guidelines) verstärken diese Sensitivität UND verbessern die absolute Performance."
RQ4: "Fine-Tuning testet ob diese inhärente Sensitivität das Training überlebt oder ob Shortcuts sie zerstören."
