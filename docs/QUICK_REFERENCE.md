# Quick Reference: GAIC Unified Experiments

## 🚀 Running Experiments

### Quick Test (1 dataset, 10 samples)

```bash
# Use the test configuration
uv run gaic/unified_experiment.py config/experiments/test_config.toml
```

### Full Experiment (All datasets, 100 samples)

```bash
# Use the main configuration
uv run gaic/unified_experiment.py config/experiments/experiment_config.toml

# Or simply (uses default config)
uv run gaic/unified_experiment.py
```

### Custom Configuration

```bash
# Create your own config file
cp config/experiments/test_config.toml config/experiments/my_config.toml

# Edit it as needed
vim config/experiments/my_config.toml

# Run with your config
uv run gaic/unified_experiment.py config/experiments/my_config.toml
```

## 📊 Analyzing Results

### View Latest Results

```bash
# Analyze latest results in unified_outputs/
uv run gaic/analyze_results.py

# Or specify a different results directory
uv run gaic/analyze_results.py experiments/unified_outputs
```

### Manual Results Inspection

```bash
# List all results
ls -lth experiments/unified_outputs/

# View latest manipulation results
cat experiments/unified_outputs/manipulation_*.json | jq '.'

# View latest context results
cat experiments/unified_outputs/context_*.json | jq '.'

# Extract specific dataset
cat experiments/unified_outputs/context_*.json | jq '.datasets.ABSTRCT'
```

## ⚙️ Configuration Examples

### Test Single Dataset

```toml
[experiment]
sample_size = 10

[datasets]
enabled = ["ABSTRCT"]

[manipulation]
enabled = true

[context]
enabled = false  # Skip context for now
```

### Test All Datasets (Small Sample)

```toml
[experiment]
sample_size = 20

[datasets]
enabled = [
    "ABSTRCT", "ACQUA", "AEC", "AFS", "ARGUMINSCI",
    "FINARG", "IAM", "PE", "SCIARK", "USELEC"
]
```

### Full Production Run

```toml
[experiment]
sample_size = 100

[datasets]
enabled = [
    "ABSTRCT", "ACQUA", "AEC", "AFS", "ARGUMINSCI",
    "FINARG", "IAM", "PE", "SCIARK", "USELEC"
]

[manipulation]
enabled = true
manipulations = ["M0", "M1", "M2"]

[context]
enabled = true
levels = ["C0", "C1", "C2", "C3"]
```

### Different LLM Models

```toml
# Ollama with different model
[llm]
provider = "ollama"
model = "mistral:7b"

# Or
model = "llama3.2:3b"

# Or
model = "qwen2.5:14b"
```

## 📋 Expected Results

### Part 1: Manipulation

- **M0 (baseline)**: Best performance
- **M1 (Feger)**: Slight drop if model relies on function words
  - Feger et al. found encoder Δ ≤ 0.02
  - Your decoder should be compared to this
- **M2 (shuffle)**: Large drop (sanity check)

### Part 2: Context

- **C0 (definition only)**: Baseline
- **C1 (+ document)**: Should improve if context helps
- **C2 (+ guidelines)**: Should improve with instructions
- **C3 (+ both)**: Best performance (your MVP baseline)

## 🔍 Monitoring

### Watch Progress Live

The runner shows:

```
🔬 Processing dataset: ABSTRCT
  Testing manipulation: M0
  M0: 100%|████████████| 100/100 [01:23<00:00,  1.20it/s]
```

### Check Intermediate Results

Results are saved immediately after each experiment completes.

## 🐛 Troubleshooting

### Ollama Not Running

```bash
# Start Ollama
ollama serve

# In another terminal, verify model is available
ollama list
ollama pull llama3.1:8b  # If needed
```

### Low Performance

```toml
# Try different prompts
[prompts]
system = """You are an expert argument classifier trained on academic texts."""

# Or adjust temperature
[llm]
temperature = 0.1  # Slightly more variation
```

### Document Context Not Working

Check which datasets have document context:

```python
import json
from pathlib import Path

for ds in ["ABSTRCT", "ACQUA", "AEC", "AFS", "ARGUMINSCI",
           "FINARG", "IAM", "PE", "SCIARK", "USELEC"]:
    with open(f"context/{ds}/dataset.json") as f:
        data = json.load(f)
        has_doc = data["capabilities"]["has_document_context"]
        print(f"{ds}: {'✓' if has_doc else '✗'}")
```

## 📈 Analysis Workflow

1. **Run experiments**

   ```bash
   uv run gaic/unified_experiment.py
   ```

2. **Analyze results**

   ```bash
   uv run gaic/analyze_results.py
   ```

3. **Compare metrics**
   - Look at mean Δ_feger across datasets
   - Identify best context level (C0-C3)
   - Check which datasets benefit most from context

4. **Iterate**
   - Adjust prompts in config
   - Try different LLM models
   - Test different context strategies

## 🎯 Goals

### Part 1: Manipulation

**Research Question**: Does the decoder rely on linguistic structure?

- Compare Δ_feger to encoder baseline (≤ 0.02)
- Verify Δ_shuffle is large (sanity check)

### Part 2: Context

**Research Question**: Does available context improve zero-shot performance?

- Find best context configuration
- Measure improvement: Δ_context = F1(Cx) - F1(C0)
- **Use best config as GAIC submission baseline**

## 📁 File Structure

```
config/experiments/
  ├── experiment_config.toml    # Main production config
  └── test_config.toml           # Quick test config

gaic/
  ├── unified_experiment.py      # Main runner
  ├── analyze_results.py         # Results analyzer
  ├── document_context.py        # Document extraction
  └── helper.py                  # Manipulation functions

experiments/unified_outputs/
  ├── manipulation_YYYYMMDD_HHMMSS.json
  ├── context_YYYYMMDD_HHMMSS.json
  ├── manipulation_analysis_latest.json
  └── context_analysis_latest.json

context/
  └── {DATASET}/
      ├── dataset.json          # Capabilities & metadata
      ├── definition.md         # Argument definition
      └── guideline.md          # Annotation guidelines
```

## 💡 Tips

1. **Start small**: Test with 1 dataset and 10 samples first
2. **Run separately**: Test manipulation and context independently
3. **Monitor resources**: LLM calls take time, plan accordingly
4. **Save configs**: Keep different configs for different experiments
5. **Version results**: Results include timestamps for tracking

## 🏆 Success Criteria

- **Manipulation**: Δ_feger close to 0.02 (comparable to encoder)
- **Context**: C3 > C2 > C1 > C0 (context helps)
- **MVP**: Identify best zero-shot config for GAIC submission
