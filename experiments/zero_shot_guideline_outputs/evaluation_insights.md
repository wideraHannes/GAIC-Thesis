Only 4 out of 10 datasets have a guideline pdf.
Here we concentrated us on these 4 datasets and summerized them with gpt5.2 into a concise argument defintion per dataset.
At first `zero_shot_by_dataset_20260128_164044_llama.3.1_ARGUMINSCI_GUIDELINE` we simple put the ARGUMINSCI Guideline into the prompt and evaluated the results.
At second we passed in the correct guideline `zero_shot_with_guidelines_20260128_165207_llama.3.1_correct_Guideline`.
As expected in latter we found way better results - motivating further investigation and showing that we utilize the guideline somehow.

## Global Results Comparison

| Approach                            | Accuracy | Precision | Recall | F1 Score |
| ----------------------------------- | -------- | --------- | ------ | -------- |
| ARGUMINSCI Guideline Only           | 0.73     | 0.65      | 1.00   | 0.78     |
| Correct Dataset-Specific Guidelines | 0.85     | 0.77      | 1.00   | 0.87     |

**Sample Size:** 40 total samples (10 per dataset across 4 datasets with guidelines)
