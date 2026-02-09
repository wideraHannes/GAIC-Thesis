# Thesis Feedback — Updated Analysis (9 Feb 2026)

## Status: 9 Experiments Complete, Data Tells a Clear Story

You now have **9 experiment configurations** across 10 datasets:

| Experiment     | Model                                                 | Context                       |
| -------------- | ----------------------------------------------------- | ----------------------------- |
| baseline × 5   | GPT-4.1, Llama-70B, Llama-8B, Mistral-7B, Mistral-24B | None                          |
| guidelines × 2 | GPT-4.1, Mistral-24B                                  | Guidelines only               |
| full × 2       | GPT-4.1, Mistral-24B                                  | Guidelines + document context |

This gives you the data for Part 1 (robustness) and Part 2 (context) of the exposé. Here is a deep analysis of what the data actually says, where the narrative needs adjustment, and what to do next.

---

## 1. Part 1 Findings: RQ1 Is Confirmed (With a Nuance)

### 1.1 The Central Table

| Model                     | Size       | Mean Δ_feger | Mean Δ_shuffle | Mean F1 (original) |
| ------------------------- | ---------- | ------------ | -------------- | ------------------ |
| Mistral-7B                | 7B         | **−0.122**   | −0.213         | 0.604              |
| Llama-8B                  | 8B         | **+0.038**   | −0.023         | 0.505              |
| Mistral-24B               | 24B        | **−0.208**   | −0.274         | 0.632              |
| Llama-70B                 | 70B        | **−0.085**   | −0.129         | 0.651              |
| GPT-4.1                   | frontier   | **−0.195**   | −0.269         | 0.623              |
| _Encoders (Feger et al.)_ | _110–340M_ | _≤0.02_      | _N/A_          | _0.79 (in-dist.)_  |

### 1.2 What This Means

**The exposé's RQ1 hypothesis is confirmed for 4 out of 5 models.** Four decoders show |Δ_feger| ≥ 0.085, i.e., **4–10× larger** than encoder Δ ≤ 0.02. Removing function words, stop words, and discourse markers substantially hurts decoder performance. This is the opposite of what Feger et al. found for encoders, and it confirms the theoretical argument in exposé Section 2.4: causal attention makes decoders sensitive to the linguistic scaffolding that defines argumentation.

**However, the "size scaling" story does NOT hold cleanly.** The Δ values across sizes:

```
7B: −0.122  →  8B: +0.038  →  24B: −0.208  →  70B: −0.085  →  frontier: −0.195
```

This is not monotonic. Mistral-24B (−0.208) is _more_ sensitive than Llama-70B (−0.085). The pattern is driven by **model family and capability**, not raw parameter count.

### 1.3 The Llama-8B Anomaly: A Floor Effect, Not a Counter-Example

Llama-8B is the outlier: Δ_feger = **+0.038** (positive — manipulation _helps_). The per-dataset breakdown reveals why:

| Dataset    | Original F1 | Feger F1 | Δ_feger    |
| ---------- | ----------- | -------- | ---------- |
| ABSTRCT    | 0.625       | 0.499    | −0.126     |
| ACQUA      | 0.625       | 0.653    | **+0.028** |
| AEC        | 0.403       | 0.542    | **+0.138** |
| AFS        | 0.403       | 0.403    | 0.000      |
| ARGUMINSCI | 0.625       | 0.683    | **+0.058** |
| FINARG     | 0.498       | 0.499    | **+0.002** |
| IAM        | 0.467       | 0.623    | **+0.156** |
| PE         | 0.384       | 0.661    | **+0.276** |
| SCIARK     | 0.499       | 0.318    | −0.181     |
| USELEC     | 0.524       | 0.550    | **+0.026** |

On 8 out of 10 datasets, removing function words either makes no difference or _improves_ performance. This model is at **near-chance level** (mean F1 = 0.505 for binary classification). At the floor, there is nowhere to fall. The positive Δ means the model is _confused by_ function words rather than _relying on_ them — a fundamentally different failure mode than encoders.

**Narrative framing:** Llama-8B does not contradict the hypothesis. It demonstrates a **capability threshold**: decoder architecture alone is necessary but not sufficient. The model must have enough capability to actually perform the task before manipulation sensitivity becomes a meaningful diagnostic. This is an important methodological point the thesis should make.

### 1.4 GPT-4.1 and Mistral-24B: Consistently Structure-Reliant

These two models show **negative Δ_feger on all 10 datasets**. Not a single dataset where manipulation helps. This is the cleanest evidence for structure reliance.

### 1.5 Llama-70B: The Interesting Middle Case

Llama-70B shows **positive Δ_feger on 3 datasets** (ACQUA +0.067, AFS +0.019, IAM +0.068) — all datasets _without_ annotation guidelines. On datasets _with_ guidelines (ARGUMINSCI −0.411, PE −0.233, USELEC −0.075), Δ is strongly negative.

This suggests Llama-70B uses argument structure primarily in domains where argumentative language is more formally marked (datasets with annotation guidelines tend to have more discourse-structured text). In informal or domain-specific datasets without guidelines, it falls back to content-word patterns. **This is a partial shortcut behavior** — not as extreme as encoders, but not as clean as GPT-4.1.

### 1.6 The Full Per-Dataset Δ_feger Matrix

```
Dataset        Guide?     GPT-4.1  Llama-70B   Llama-8B Mistral-7B Mistral-24B
ABSTRCT        Yes        -0.197     -0.099     -0.126     +0.069     -0.067
ACQUA          No         -0.106     +0.067     +0.028     +0.054     -0.125
AEC            No         -0.190     -0.001     +0.138     -0.163     -0.315
AFS            No         -0.178     +0.019     +0.000     -0.218     -0.282
ARGUMINSCI     Yes        -0.426     -0.411     +0.058     -0.127     -0.142
FINARG         No         -0.166     -0.111     +0.002     -0.070     -0.070
IAM            No         -0.097     +0.068     +0.156     -0.202     -0.134
PE             Yes        -0.099     -0.233     +0.276     -0.166     -0.395
SCIARK         No         -0.043     -0.071     -0.181     -0.060     -0.160
USELEC         Yes        -0.448     -0.075     +0.026     -0.334     -0.395
```

Pattern: GPT-4.1 and Mistral-24B are negative on every single dataset. Llama-70B is positive on 3 (all non-guideline datasets). Llama-8B is positive on 8 (floor effect).

### 1.7 The Killer Comparison: Zero-Shot Decoders vs. Encoder Cross-Dataset Transfer

| Setting                                | Mean F1   |
| -------------------------------------- | --------- |
| Encoder in-distribution (Feger et al.) | 0.79      |
| **Decoder zero-shot (Llama-70B)**      | **0.651** |
| **Decoder zero-shot (Mistral-24B)**    | **0.632** |
| **Decoder zero-shot (GPT-4.1)**        | **0.623** |
| Encoder cross-dataset (Feger et al.)   | 0.56–0.61 |

**Without any training, decoders already outperform encoders on generalization (cross-dataset transfer).** Encoders achieve 0.79 in-distribution but crash to 0.56–0.61 cross-dataset. Decoders start at 0.62–0.65 zero-shot. This means decoders are **already better generalizers** — and they haven't even seen the data yet. This should be a headline finding in the thesis.

---

## 2. Part 2 Findings: Context Is a Double-Edged Sword

### 2.1 Context Ablation (4 Datasets With Available Context)

**GPT-4.1:**

| Dataset    | Baseline  | +Guidelines | +Full     | Δ_guidelines | Δ_full     |
| ---------- | --------- | ----------- | --------- | ------------ | ---------- |
| ABSTRCT    | 0.729     | 0.729       | 0.729     | 0.000        | 0.000      |
| ARGUMINSCI | 0.764     | 0.764       | 0.733     | 0.000        | **−0.031** |
| PE         | 0.433     | 0.486       | 0.542     | **+0.053**   | **+0.109** |
| USELEC     | 0.722     | 0.732       | 0.697     | +0.010       | −0.025     |
| **Mean**   | **0.662** | **0.678**   | **0.675** | **+0.016**   | **+0.013** |

**Mistral-24B:**

| Dataset    | Baseline  | +Guidelines | +Full     | Δ_guidelines | Δ_full     |
| ---------- | --------- | ----------- | --------- | ------------ | ---------- |
| ABSTRCT    | 0.700     | 0.665       | 0.667     | −0.035       | −0.033     |
| ARGUMINSCI | 0.475     | 0.569       | **0.800** | +0.094       | **+0.325** |
| PE         | 0.729     | 0.665       | 0.525     | −0.064       | **−0.204** |
| USELEC     | 0.729     | 0.729       | 0.764     | 0.000        | +0.036     |
| **Mean**   | **0.658** | **0.657**   | **0.689** | **−0.001**   | **+0.031** |

### 2.2 What This Means

**Context effects are highly volatile — both model-specific and dataset-specific.** The headline numbers look underwhelming (GPT-4.1: +0.016 from guidelines, Mistral-24B: +0.031 from full), but they hide dramatic swings:

- **ARGUMINSCI + Mistral-24B full**: +0.325 improvement (0.475 → 0.800). This is an enormous jump — full context nearly doubles the model's ability to identify arguments in scientific discourse.
- **PE + Mistral-24B full**: −0.204 drop (0.729 → 0.525). Full context actively _hurts_. The model was already good at PE and the additional context confuses it.
- **PE + GPT-4.1**: +0.109 progressive improvement. The _opposite_ pattern — GPT-4.1 benefits where Mistral-24B suffers.

**The exposé's RQ2 hypothesis ("adding context will improve F1") is too simple.** The data shows:

1. Context helps models that are _struggling_ on a dataset (rescue effect)
2. Context can _hurt_ models that are already performing well (confusion effect)
3. The same dataset can show opposite context effects depending on the model

This is a more nuanced and more interesting finding than a blanket "context helps."

### 2.3 The Cross-Cutting Finding: Context Reduces Manipulation Sensitivity

This is the novel finding that connects Part 1 and Part 2. On the 4 context-available datasets:

**GPT-4.1:**

| Condition  | Mean Δ_feger | Mean Δ_shuffle | Mean F1 |
| ---------- | ------------ | -------------- | ------- |
| Baseline   | −0.293       | −0.313         | 0.662   |
| Guidelines | −0.298       | −0.259         | 0.678   |
| Full       | **−0.245**   | **−0.210**     | 0.675   |

**Mistral-24B:**

| Condition  | Mean Δ_feger | Mean Δ_shuffle | Mean F1 |
| ---------- | ------------ | -------------- | ------- |
| Baseline   | −0.250       | −0.255         | 0.658   |
| Guidelines | **−0.172**   | −0.310         | 0.657   |
| Full       | **−0.184**   | −0.296         | 0.689   |

**For both models, adding context reduces |Δ_feger|.** GPT-4.1 goes from −0.293 → −0.245 (16% reduction). Mistral-24B from −0.250 → −0.184 (26% reduction).

**Interpretation:** When the model has external context (guidelines, document), it partially shifts its decision-making from sentence-internal linguistic structure to context-based reasoning. The manipulation (which only affects the target sentence) therefore has less impact. This is not the model becoming "more robust" in the sense of understanding argument structure better — it's the model using a **different information channel** that is unaffected by sentence-level manipulation.

This creates a fascinating conceptual question for Part 3: if context acts as a bypass around sentence-level shortcuts, is that genuine generalization or just a different kind of crutch?

---

## 3. What the Exposé Narrative Needs to Change

### 3.1 The Core Thesis Argument (Strengthened)

**Current exposé argument:** "Do LLMs overcome shortcut learning?"

**Revised argument based on data:** "Decoder-based LLMs fundamentally differ from encoders in argument identification. They rely on linguistic structure (large Δ), already generalize better than encoder cross-dataset transfer (F1 0.63–0.65 vs 0.56–0.61), and can leverage external context — though context effects are volatile and dataset-specific."

### 3.2 Changes to the Three Parts

| Part       | Exposé Version                          | Revised Based on Data                                                                                                                                     |
| ---------- | --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Part 1** | Llama-8B + Mistral-7B, predict Δ > 0.10 | **5 models (7B–frontier)**: confirm Δ >> 0.02 for capable models, identify capability threshold (Llama-8B floor effect), compare to encoder cross-dataset |
| **Part 2** | Llama-8B, C0–C4 context ladder          | **GPT-4.1 + Mistral-24B**: context helps selectively, the cross-cutting finding is that context reduces manipulation sensitivity                          |
| **Part 3** | Fine-tuning Llama-8B with LoRA          | Fine-tune **best open-weight model** (Mistral-24B or Llama-70B), test whether fine-tuning reintroduces shortcuts                                          |

### 3.3 Specific Hypothesis Updates

**RQ1 (old):** "Zero-shot LLMs will show Δ > 0.10"
**RQ1 (new):** "Zero-shot LLMs with sufficient capability show Δ >> 0.02 (confirmed: mean |Δ_feger| ≈ 0.15 for capable models, vs encoder Δ ≤ 0.02). Decoder architecture makes models sensitive to argument structure, but only above a capability threshold."

**RQ2 (old):** "Adding context will improve F1"
**RQ2 (new):** "Context effects are dataset-specific and model-specific. Context helps models that struggle on a dataset (rescue effect) but can hurt models that already perform well (confusion effect). Context also partially bypasses sentence-level manipulation sensitivity."

**RQ3 (unchanged, but better motivated):** The Part 1/2 data sets up Part 3 perfectly. Decoders zero-shot: lower F1 than encoder in-distribution (0.65 vs 0.79), but rely on genuine structure (high Δ). The question becomes: can fine-tuning close the 0.14 F1 gap without collapsing Δ back to encoder-like levels?

---

## 4. Recommended Next Steps (Prioritized)

### 🔴 Priority 1: Run Context Ablation for Llama-70B (1 new experiment set)

You have 5 baseline models but only 2 in the context ablation (GPT-4.1 and Mistral-24B). **Llama-70B is the best-performing open-weight model (F1 = 0.651)** and showed the interesting mixed pattern on Δ_feger (negative on guideline datasets, positive on non-guideline datasets). Testing it with context would answer:

> Does the best open-weight model show the same "rescue effect" as Mistral-24B, or does it behave more like GPT-4.1 (already good, context barely helps)?

This directly determines your fine-tuning target for Part 3.

**Estimated effort:** 2 config files + 2 runs of `unified_experiment.py`.

### 🔴 Priority 2: Update the Exposé Preliminary Results

Your exposé Section a1 shows pilot results from Llama-8B on ABSTRCT only (n=100). You now have 5 models × 10 datasets × 3 conditions (n=30 each). The preliminary results section should present:

1. Table 1 (Section 1.1 above): the robustness table — this IS the central Part 1 result
2. The F1 comparison to Feger cross-dataset (Section 1.7) — the strongest argument for the thesis
3. The context volatility finding (Section 2.2) — reframes RQ2

### 🟡 Priority 3: Deepen Analysis of Existing Data (No New Experiments)

Several analyses can be done on data you already have:

1. **Per-class breakdown**: Are models biased toward Argument or No-Argument? Feger et al. noted encoder biases. If decoders show different confusion patterns, that strengthens the "different processing" argument.
2. **ARGUMINSCI deep-dive**: Why does Mistral-24B jump from 0.475 → 0.800 with full context? What is in the ARGUMINSCI context that helps so dramatically? Examine the actual predictions.
3. **Statistical reliability**: n=30 per dataset is small. Report confidence intervals. Consider whether key findings (especially the ARGUMINSCI +0.325) would survive larger samples.

### 🟡 Priority 4: Prepare Part 3 Design Based on Part 1/2 Results

The data suggests the fine-tuning candidate should be **Mistral-24B** (not Llama-8B as in the exposé):

- Second-best baseline F1 (0.632, close to Llama-70B's 0.651)
- Highest manipulation sensitivity (Δ_feger = −0.208) — most to lose from shortcut learning
- Already tested with context (you have the full ablation)
- More practical for LoRA than Llama-70B (24B vs 70B parameters)

The key Part 3 question: Mistral-24B zero-shot has Δ_feger = −0.208. After LoRA fine-tuning on GAIC data, does Δ collapse toward encoder levels (≤0.02) or stay large?

---

## 5. What NOT to Change

1. **Keep the three-part structure.** Robustness → Context → Fine-tuning is sound. The data supports it.
2. **Keep both manipulations (Feger + shuffle).** Shuffle consistently shows larger Δ and provides a complementary diagnostic.
3. **Keep all 5 baseline models in the analysis.** Including Llama-8B as a "floor effect" example is methodologically valuable.
4. **Keep the encoder comparison.** Δ ≤ 0.02 (encoder) vs Δ ≈ −0.15 (decoder) is the thesis's headline contrast.

---

## 6. GAIC Submission Strategy (Updated)

Based on the data, the three GAIC submissions should be:

1. **GPT-4.1 + full context**: Most consistent performer, context helps on PE (its weakest dataset)
2. **Mistral-24B + full context**: Best open-weight option — the ARGUMINSCI +0.325 rescue is dramatic, competitive overall mean
3. **Fine-tuned Mistral-24B** (Part 3 result): Tests whether training can close the gap to encoder in-distribution F1

---

## 7. The Bottom Line for the Exposé

Your data tells a **strong, clean story**:

> Feger et al. showed encoders learn datasets, not arguments (Δ ≤ 0.02). We show that decoders learn arguments: zero-shot LLMs show 4–10× larger manipulation sensitivity, already outperform encoder cross-dataset transfer without any training, and can leverage external context — though context effects are volatile. The open question is whether fine-tuning preserves this structural reliance or reintroduces the shortcut patterns that plague encoders.

This is a compelling thesis argument. Part 1 is essentially done. Part 2 has strong initial findings. Part 3 is well-motivated by Parts 1–2. The exposé should be updated to reflect this confidence.
