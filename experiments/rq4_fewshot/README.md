# RQ4: Few-Shot Prompting for Argument Identification

## Research Question

Does in-context learning (ICL) with few-shot examples improve argument identification beyond zero-shot prompting with context (C1: definition)?

## Experimental Setup

**Baseline**: Zero-shot with C1 context (definition only)

**Few-shot strategies tested**:
1. **Deterministic**: First k Arguments + first k No-Arguments from dataset's training split
2. **Retrieval-based**: Top-k most similar examples per label using embedding similarity (ChromaDB + text-embedding-3-small)

**Prompt structure** (final version):
```
## Examples

### Sentences that ARE Arguments:
1. "..."
2. "..."
3. "..."

### Sentences that are NOT Arguments:
1. "..."
2. "..."
3. "..."
```

## Results Summary

### Deterministic Few-Shot (IAM, n=60)

| Model | k=0 (baseline) | k=3 | k=5 | k=7 |
|-------|----------------|-----|-----|-----|
| GPT-5.2 | 0.7494 | — | — | **0.7818** (+0.03) |
| Mistral-medium | 0.7643 | — | — | **0.6677** (-0.10) |

**Finding**: Few-shot helps GPT-5.2 marginally but *hurts* Mistral significantly.

### k-NN Baseline (no LLM, embedding similarity + majority voting)

| Method | Mean F1 (10 datasets) |
|--------|----------------------|
| k=50 simple voting | 0.6619 |
| k=50 weighted voting | 0.6616 |
| k=50 per-dataset only | TBD |

**Finding**: Embedding-based classification achieves ~0.66 F1, below LLM zero-shot (~0.75). High variance across datasets (ABSTRCT: 0.87, AEC: 0.49).

## Key Insights

1. **Model sensitivity**: Few-shot effectiveness is model-dependent. Arbitrary examples can introduce noise that degrades performance (Mistral collapse).

2. **Example selection matters more than quantity**: The Mistral failure suggests that *which* examples are shown matters more than *how many*.

3. **Embedding space limitations**: k-NN baseline shows embeddings partially encode argumentativeness but with high dataset variance, suggesting domain/annotation schema differences dominate the embedding space.

4. **Retrieval did not rescue performance**: Similarity-based example selection did not consistently outperform deterministic selection or zero-shot baselines.

## Conclusion

Few-shot prompting does not reliably improve argument identification over zero-shot with definitions. The marginal gains (GPT-5.2: +0.03) do not justify the added complexity, while the risk of degradation (Mistral: -0.10) is significant.

**Recommendation**: For production use, prefer zero-shot with C1 context (definition). Few-shot adds complexity without consistent benefit.

## Files

- `config/experiments/rq4_fewshot/` — deterministic few-shot configs
- `config/experiments/rq4_retrieval/` — retrieval-based few-shot configs
- `experiments/rq4_retrieval/knn_baseline.ipynb` — k-NN baseline analysis
- `gaic/embeddings.py` — ChromaDB vector store for retrieval
