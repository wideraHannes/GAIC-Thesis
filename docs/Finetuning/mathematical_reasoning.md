### 1. The Math Behind a Classification Head (The Old Way)

In my ACL 2025 paper, we used encoders like BERT and RoBERTa. These models use a classification head.

Mathematically, the model encodes the input sentence $x$ into a dense vector representation, usually the [CLS] token: $h_{\text{[CLS]}}$. This vector is then passed through a randomly initialized linear layer (the classification head) with weights $W$ and bias $b$, followed by a softmax function to get the probability of the class $y$ (Argument or No-Argument):
$$P(y \mid x; \theta) = \text{Softmax}(W \cdot h_{\text{[CLS]}} + b)$$

The model is optimized using standard Cross-Entropy Loss:
$$\mathcal{L}_{\text{clf}} = - \sum_{c \in C} y_c \log P(y_c \mid x; \theta)$$

**The Problem (Shortcut Learning):** Because the classification head $W$ directly maps the sentence representation $h_{\text{[CLS]}}$ to a fixed label space $C$, the network will take the path of least resistance to minimize $\mathcal{L}_{\text{clf}}$. If the word "Therefore" appears in 80% of the "Argument" training examples, the attention mechanism just assigns a massive positive weight to "Therefore" in $h_{\text{[CLS]}}$, and $W$ maps that directly to the "Argument" class. It completely bypasses reading the sentence logically. This is why the encoders failed our manipulation tests ($\Delta \le 0.02$).

### 2. The Math Behind GoLLIE Training (Our Way)

In our decoder setup (Ministral 8B/14B), there is no classification head. We are treating argument identification as an auto-regressive next-token prediction task, conditioned on both the input sentence $x$ and the dynamic guideline context $g$.

The probability of generating the target label sequence $y$ (e.g., the word "Argument") of length $N$ tokens is:
$$P(y \mid x, g; \theta) = \prod_{i=1}^{N} P(y_i \mid y_{<i}, x, g; \theta)$$

The model minimizes the causal language modeling loss:
$$\mathcal{L}_{\text{gen}} = - \sum_{i=1}^{N} \log P(y_i \mid y_{<i}, x, g; \theta)$$

### Why the Dynamic Context is Mathematically Mandatory

Here is where your idea to use 10-20 paraphrases saves the thesis.

If you use the _exact same_ guideline $g$ for every single training step in ABSTRCT, $g$ becomes a mathematical constant. From an optimization standpoint, the model's parameters $\theta$ will realize that calculating attention over $g$ is a waste of compute because it never changes. The model will mathematically collapse the probability distribution to ignore $g$:
$$P(y_i \mid y_{<i}, x, g; \theta) \approx P(y_i \mid y_{<i}, x; \theta)$$

When that collapse happens, your causal decoder has essentially reverted back into a standard classification head! It will start looking for shortcuts in $x$ again.

By introducing lexical variance (paraphrasing the guidelines) and structural randomization, $g$ becomes a highly variable tensor at every training step. The model _cannot_ minimize $\mathcal{L}_{\text{gen}}$ without actively routing its attention through $g$ to figure out what the rules are for this specific batch.

### The Benefit vs. Classification Heads

1. **Zero-Shot Generalization:** A classification head locks the model to a fixed number of classes explicitly seen during training. Our GoLLIE-style decoder learns a _function_ (how to apply rules $g$ to text $x$). This is why it can evaluate completely unseen datasets with novel guidelines in the CLEF 2026 evaluation dataset.
2. **Shortcut Resistance:** A classification head optimizes for feature-label correlations. A text-to-text decoder conditioned on varying rules optimizes for reading comprehension.
