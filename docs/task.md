# Generalizability of Argument Identification in Context 2026

## Task

Given a sentence from a dataset along with metadata about its provenance, such as the source text and the dataset's annotation guidelines, predict whether the sentence can be annotated as an argument or not. In particular, participants are encouraged to develop robust systems that generalize beyond lexical shortcuts to unseen datasets and investigate ways to exploit rich context information for this purpose.

## Data

For the task, training data will be provided as a subset of the 17 benchmark datasets totaling about 345k labeled sentences and identified as most relevant for argument identification in this paper. This subset includes sentences each labeled as argument or no-argument, according to the respective dataset annotations, along with accompanying metadata such as IDs, generated training and development splits, links to original data sources and annotation guidelines, as well as the scripts used for data preparation.

## Evaluation

The systems will be evaluated on test data that differs from the development data. This includes partially or fully held-out portions of the datasets used for sampling, as well as newly created data reflecting diverse domains and annotation guidelines. This setup addresses the risk of data contamination in LLMs and for participants’ potential use of additional datasets during training. Generalizability will be measured using the macro F
-score. To evaluate the systems, the macro F
-score will be specified for each test dataset, along with the overall average of all these values.

## Submission

We ask participants to use TIRA for submissions. Each team can submit up to 3 approaches to the task.

The submissions for this task must be made as a run submission, meaning the test data will be provided in the same format as the training and development data, and participants must return their predictions.

### Output Format

The output of the submission needs to be a JSONL file. Each line in the file must be in the following JSON format:

id: The ID of the sentence that was classified.
label: The label assigned by your classifier (Argument if the sentence is an argument and No-Argument otherwise).
Example JSONL file (click to see)

### Input Format

The input for the submission will also be provided as a JSONL file, where each line follows the JSON structure below:

id: A unique sentence identifier composed of a dataset prefix, the split name, and a running number (e.g., ABSTRCT-train-1).
paper: A link to the corresponding dataset paper.
document: A link to the source document from which the sentence was extracted.
guidelines: A link to the annotation guidelines used to label the sentence.
label: The gold label, where Argument indicates an argument sentence and No-Argument otherwise.
sentence: The sentence itself.

## Related Work

- Marc Feger, Katarina Boland, and Stefan Dietze. Limited Generalizability in Argument Mining: State-Of-The-Art Models Learn Datasets, Not Arguments. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics, July 2025.
- Terne Sasha Thorn Jakobsen, Maria Barrett, and Anders Søgaard. Spurious Correlations in Cross-Topic Argument Mining. In Proceedings of \*SEM 2021: The Tenth Joint Conference on Lexical and Computational Semantics, August 2021.
- Robert Geirhos, Jörn-Henrik Jacobsen, Claudio Michaelis, Robert Zemel, Wieland Brendel, Matthias Bethge & Felix A. Wichmann. Shortcut learning in deep neural networks. Nature Machine Intelligence, November 2020.
