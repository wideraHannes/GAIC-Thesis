# Experiment Plan: Novel AI Experiments for GAIC Thesis

## Context
Building on initial investigation showing LLMs utilize guidelines (85% accuracy with correct vs 73% with wrong guidelines). User interested in fine-tuning approaches (LoRA, DPO). Advisor recommends caution: fine-tuning should diagnose shortcut learning, not just improve performance.

---

## PRIORITY 0: Fix the 100% Recall Bias

**Issue:** Model predicts "Argument" too often (catches all arguments but many false positives).

**Cause identified:** Prompt formulation issue.

**Action required before any other experiments:**
- Analyze why model defaults to "Argument"
- Test alternative prompt formulations
- Balance precision/recall before scaling up

---

## Experiment 1: Shortcut Injection Study (Diagnostic)

**Pitch:** "Directly measuring LLM shortcut reliance through controlled corruption"

**Why this is smart:**
- Uses fine-tuning as diagnostic tool, not just performance improvement
- Directly answers RQ1 (do LLMs rely on shortcuts?)
- Produces clear, publishable findings

**Implementation:**

**Phase 1 - Identify shortcuts:**
Use existing `marker_correlations.csv` data showing lexical-label correlations:
- "therefore" → Argument (PE: 0.125)
- "however" → Argument (ARGUMINSCI: 0.120)

**Phase 2 - Create corrupted dataset:**
```python
# Inject artificial lexical cues that are perfectly predictive
for sample in test_set:
    if sample.label == "Argument":
        sample.text = sample.text + " Therefore, this is clear."
    else:
        sample.text = sample.text + " This is merely descriptive."
```

**Phase 3 - Compare:**
| Condition | Clean Data | Corrupted Data |
|-----------|-----------|----------------|
| Zero-shot | X% | Y% |
| Zero-shot + guidelines | X% | Y% |
| Fine-tuned | X% | Y% |

**What results reveal:**
- Accuracy increase on corrupted = model relies on shortcuts
- Larger increase after fine-tuning = fine-tuning amplified shortcuts
- Guidelines reduce corruption benefit = guidelines help resist shortcuts

**Files to create:** `gaic/shortcut_injection.py`

---

## Experiment 2: Leave-One-Dataset-Out Evaluation

**Pitch:** "Does learning from 9 datasets transfer to the 10th?"

**Two variants:**

### Variant A: Zero-Shot (no fine-tuning needed)
Test cross-domain generalization with current setup:
```
For each dataset D:
    Evaluate zero-shot on D
    Compare: with generic prompt vs. with D's guideline vs. with wrong guideline
```

### Variant B: LoRA Fine-Tuning (if time permits)
```
For each dataset D:
    Fine-tune on 9 other datasets
    Evaluate on D
    Compare to zero-shot on D
```

**Key ablation for Variant B:**
- Fine-tune WITHOUT guidelines in training
- Fine-tune WITH guidelines in training
- Test: Does training with guidelines improve cross-dataset generalization?

**What results reveal:**
- Fine-tuning improves held-out = learning transferable argumentation
- Fine-tuning hurts held-out = overfitting to dataset shortcuts
- High variance = shortcut reliance varies by domain

**Files to create:** `gaic/leave_one_out.py`, `gaic/lora_finetune.py`

---

## Experiment 3: Reasoning Chains (Chain-of-Thought)

**Pitch:** "Can we teach LLMs to reason like annotators?"

**Implementation:**
```python
REASONING_PROMPT = """
## Guidelines
{guidelines}

## Sentence
{sentence}

## Instructions
1. Quote the specific criterion from guidelines most relevant
2. Explain how sentence satisfies/violates this criterion
3. State your conclusion

RELEVANT CRITERION: [quote]
REASONING: [analysis]
CONCLUSION: [Argument/No-Argument]
"""
```

**Analysis:**
- Parse reasoning for guideline citations
- Compute "reasoning fidelity score"
- Test if CoT improves generalization

**Files to create:** `gaic/reasoning_chains.py`

---

## Experiment 4: Adversarial Probing with Attention Analysis

**Pitch:** "Mechanistic evidence for shortcut reliance through attention patterns"

**Why this is Master-level:**
- Combines mechanistic interpretability (hot topic in AI safety research)
- Requires custom tooling - not just running existing scripts
- Directly visualizes what the model is "looking at"
- Creates compelling visualizations for thesis

**Core hypothesis:**
If LLMs use shortcuts, their attention patterns should reveal this. When predicting "Argument", does the model attend to:
- Content words (actual reasoning indicators) → good generalization
- Lexical triggers like "therefore", "however" → shortcut reliance

**Implementation:**

**Phase 1 - Attention Extraction Pipeline:**
```python
# Extract attention weights during inference
from transformers import AutoModelForCausalLM, AutoTokenizer

def extract_attention(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs, output_attentions=True)
    # Returns tuple of attention tensors per layer
    # Shape: (batch, heads, seq_len, seq_len)
    return outputs.attentions
```

**Phase 2 - Token Classification:**
```python
SHORTCUT_TOKENS = ["therefore", "however", "thus", "hence", "clearly",
                   "obviously", "argue", "claim", "believe"]
CONTENT_TOKENS = [...]  # Domain-specific content words per dataset

def compute_attention_mass(attentions, token_ids, shortcut_ids, content_ids):
    """Compute % attention on shortcut vs content tokens"""
    shortcut_attention = attentions[:, :, :, shortcut_ids].sum()
    content_attention = attentions[:, :, :, content_ids].sum()
    return shortcut_attention / (shortcut_attention + content_attention)
```

**Phase 3 - Comparative Analysis:**
| Condition | Attention on Shortcuts | Attention on Content |
|-----------|----------------------|---------------------|
| Clean data | X% | Y% |
| Corrupted data (Exp 1) | X% | Y% |
| With guidelines | X% | Y% |

**Phase 4 - Visualizations:**
- Attention heatmaps per dataset
- Layer-wise attention distribution plots
- Cross-dataset attention pattern comparison

**What results reveal:**
- High shortcut attention = model relies on lexical cues
- Guidelines shift attention to content = guidelines help resist shortcuts
- Attention patterns differ by dataset = domain-specific shortcut reliance

**Files to create:** `gaic/attention_analysis.py`, `gaic/attention_viz.py`

---

## Experiment 5: Contrastive Probing Classifier

**Pitch:** "Where in the network does the model encode argumentativeness?"

**Why this is Master-level:**
- Proper representation learning analysis
- Requires understanding of transformer layer semantics
- Trains custom classifiers on internal representations
- Publishable methodology in interpretability research

**Core hypothesis:**
Different layers encode different features. If shortcuts dominate:
- Early layers: encode surface lexical patterns (shortcuts)
- Middle layers: encode syntactic structure
- Late layers: encode task-relevant semantic features

**Implementation:**

**Phase 1 - Hidden State Extraction:**
```python
def extract_hidden_states(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs, output_hidden_states=True)
    # Returns tuple of hidden states per layer
    # Shape: (batch, seq_len, hidden_dim)
    return outputs.hidden_states  # List of [layer_0, layer_1, ..., layer_n]
```

**Phase 2 - Probing Classifier Training:**
```python
from sklearn.linear_model import LogisticRegression

def train_probing_classifier(hidden_states_per_layer, labels):
    """Train separate classifier for each layer"""
    results = {}
    for layer_idx, hidden_states in enumerate(hidden_states_per_layer):
        # Use [CLS] or mean pooling
        pooled = hidden_states.mean(dim=1)  # (batch, hidden_dim)

        clf = LogisticRegression(max_iter=1000)
        clf.fit(pooled, labels)
        results[layer_idx] = clf.score(pooled, labels)
    return results
```

**Phase 3 - Shortcut Feature Analysis:**
```python
def analyze_shortcut_encoding(model, clean_data, corrupted_data):
    """Compare which layer encodes shortcut vs. content features"""
    clean_states = extract_hidden_states(model, clean_data)
    corrupt_states = extract_hidden_states(model, corrupted_data)

    # Train classifier on clean, test on corrupted
    # If early layers transfer well → shortcuts encoded early
    # If late layers transfer poorly → late layers encode content
```

**Phase 4 - Guideline Impact Analysis:**
Compare hidden state distributions:
- Without guidelines
- With correct guidelines
- With wrong guidelines

Use techniques like:
- CKA (Centered Kernel Alignment) similarity
- Representation similarity analysis (RSA)
- t-SNE/UMAP visualization of hidden states

**What results reveal:**
- Layer X has highest probing accuracy = argumentativeness encoded at layer X
- Early layers transfer on corrupted data = shortcuts encoded early
- Guidelines shift late-layer representations = guidelines affect semantic encoding

**Visualization outputs:**
- Layer-wise probing accuracy curves
- t-SNE plots of hidden states colored by label
- Representation shift heatmaps (with/without guidelines)

**Files to create:** `gaic/probing_classifier.py`, `gaic/representation_analysis.py`

---

## NOT Recommended: DPO Fine-Tuning

**Why skip:**
- Need preference pairs showing "good" vs "bad" reasoning
- But defining "good reasoning" encodes our assumptions
- Tests whether DPO CAN reduce shortcuts, not whether shortcuts EXIST
- Does not directly answer thesis RQs

---

## Prioritized Execution Order

### Phase 1: Foundation (Weeks 1-2)
1. **Fix 100% recall bias** - prompt engineering
2. **Scale up samples** - from 10 to 50-100 per dataset
3. **Complete 4×4 guideline matrix** - each guideline applied to each dataset

### Phase 2: Core Experiments (Weeks 3-6)
4. **Experiment 1: Shortcut Injection** - diagnostic study
5. **Experiment 2A: Leave-One-Out Zero-Shot** - cross-domain baselines

### Phase 3: Mechanistic Interpretability (Weeks 7-10)
6. **Experiment 4: Attention Analysis** - visualize what the model attends to
7. **Experiment 5: Probing Classifiers** - layer-wise representation analysis

### Phase 4: Advanced (Weeks 11-14, if time permits)
8. **Experiment 3: Reasoning Chains** - interpretability via CoT
9. **Experiment 2B: LoRA Fine-Tuning** - with/without guidelines ablation

---

## Critical Files to Modify/Create

| File | Purpose |
|------|---------|
| `gaic/zero_shot_with_guidelines.py` | Fix recall bias in prompts |
| `gaic/shortcut_injection.py` | NEW: Experiment 1 |
| `gaic/leave_one_out.py` | NEW: Experiment 2A |
| `gaic/attention_analysis.py` | NEW: Experiment 4 - attention extraction |
| `gaic/attention_viz.py` | NEW: Experiment 4 - heatmap visualizations |
| `gaic/probing_classifier.py` | NEW: Experiment 5 - layer-wise probes |
| `gaic/representation_analysis.py` | NEW: Experiment 5 - CKA, t-SNE, RSA |
| `gaic/reasoning_chains.py` | NEW: Experiment 3 |
| `gaic/lora_finetune.py` | NEW: Experiment 2B (optional) |

---

## Verification Plan

For each experiment:
1. Run on 4 guideline datasets first (ABSTRCT, ARGUMINSCI, PE, USELEC)
2. Validate on remaining 6 datasets
3. Statistical testing: 3 seeds, report mean ± std
4. Visualizations for thesis figures

---

## Key Question to Address

**Before proceeding:** Analyze the 100% recall bias more deeply.
- Why does model default to "Argument"?
- Which prompt phrasings trigger this?
- What linguistic features in test sentences correlate with false positives?

This error analysis may be more valuable than fine-tuning experiments.
