### 1. Grounding the Problem: Shortcut Learning and Generalizability

First, you need to validate that the problem we are solving is recognized by the broader community, not just by our lab.

- **The Literature:** A recent paper at COLING 2025, _"Looking at the Unseen: Effective Sampling of Non-Related Propositions for Argument Mining,"_ explicitly tackles how models fail to learn relevant features and instead become biased. They note that "recent findings highlight the limitations of argument mining systems in terms of generalisability," directly echoing the core premise of my ACL 2025 paper.
- **The Thesis Angle:** You use this to justify the GAIC task. Traditional sequence labeling models (like the BERT encoders I tested) fall back on shortcuts because standard training paradigms allow them to ignore long-term context.

### 2. The Paradigm Shift: Why Decoders are the Cure

Your decision to use a generative LLM (like Ministral) instead of a classification head is backed by a massive wave of papers published just this month (March 2026).

- **The Literature:** In the newly released preprint _"Compact Prompting in Instruction-tuned LLMs for Joint Argumentative Component Detection"_ (March 2026), the authors reframe argument component detection as a language generation task. They show that instruction-tuned LLMs using compact prompts achieve higher performance than state-of-the-art pipeline systems. Similarly, _"Argument Component Segmentation with Fine-Tuned Large Language Models"_ (March 2026) demonstrates that fine-tuned, compact, openly available LLMs can achieve human-level segmentation quality. Another paper, _"End-to-End Argument Mining in Student Essays,"_ highlights how modern LLMs like Mistral and LLaMA use self-attention to efficiently capture contextual relationships, revolutionizing the field.
- **The Thesis Angle:** These papers prove that treating argument mining as a text-to-text generative task is the new state-of-the-art. However, none of them explicitly solve the _cross-dataset shortcut problem_. This is where your thesis steps in to fill the gap.

### 3. The GoLLIE Connection: Guidelines as Dynamic Context

Now we connect the generative LLM approach to the specific GoLLIE data strategy you are using.

- **The Literature:** The original GoLLIE paper (Sainz et al., ICLR 2024) proved that annotation guidelines improve zero-shot information extraction. But to bring it into 2026, we look back at that _"Compact Prompting in Instruction-tuned LLMs"_ paper. They found that instructing the LLM to jointly delimit and classify arguments via prompt instructions works incredibly well, but they also noted a severe limitation: LLMs can hallucinate or ignore constraints.
- **The Thesis Angle:** This perfectly justifies your mathematical approach (from our previous discussion). By dynamically paraphrasing the C1/C2 guidelines and using a GoLLIE approach, you force the LLM's attention mechanism to constantly read the rules. This prevents the "hallucination" and constraint-ignoring behaviors noted in the March 2026 literature, while directly answering the GAIC challenge to build systems that generalize beyond lexical shortcuts.

### How to Write This Up

In your **Chapter 2 (Background)**, create a section called _“The Shift to Generative Argument Mining.”_ Cite the March 2026 papers showing that everyone is moving to instruction-tuned decoders like Mistral.

Then, in your **Chapter 5 (Methodology/Discussion)**, state: _"While recent work (e.g., Compact Prompting, 2026) has shown the power of instruction-tuning for AM, it remains vulnerable to the shortcut learning identified by Feger et al. (2025). To address the GAIC task's requirement for cross-dataset generalization, this thesis adapts the GoLLIE framework, using dynamic context variation to prevent mathematical collapse into a classification head."_
