## KW 23

Write Notebook Paper give it Marc Latest Tuesday!

## KW 22

- Start writing the Background sections!:
  (done)-- 2.4 Decoder
  (WIP) -- 2.4.x In Context Learning
  (WIP) -- 2.5 Data Contamination LLMs

Essential:

- Results
  - Combine Formalization with Results
- Discussion
- Finish

- already Write about my Results

## KW 21

-> continue with the section why Encoder is Problematic Weaknesses and Shortcut learning via the mathematical example and other desired properties
-> define a list of desired properties for the task
-> So we have these desired properties defined q_d, g_d they are explicitly what the labeller sued to label the sentence with Y (true label) and what is required therefore to label correctly. What we are doing we take the true q_D ----> and make it q_D (summerized) and q_D (summmerized) so our system is an approximation to the system that the labeller used not a perfect one and we need to introduce Y_pred

-> Talk about failure modes in our system:
-> one Failure mode is A bad summerization extraction from the true q_D,  
-> Failed LLM to predict
-> Wrong label by annotators

## KW 20

- Define Classification as a generative Task (Decoders for argument mining)
  - https://wandb.ai/gladiator/LLMs-as-classifiers/reports/LLMs-are-machine-learning-classifiers--VmlldzoxMTEwNzUyNA

## KW 19

(done)-> Run Eval Experiments on the Heldout dataset with best model

(done)-> decide what todo with the 60 sample experiments dev

(done)-> Evaluation on Dev would be nicer

## KW 18

- Add new AEC zu tables?
  - We need to run it with the other models
- Add KNN to Baseline Comparison
- Write the Text for the results
- Integrate formalizing task and Methodology
- Zugfahrt:
  - Read fegers thesis (done)

## KW 16

--> MAYBE Experiment with Longer context from 2 -> 6 sentences prior?

## KW 15

--> Gollie Finetuning (cancled for now)
--> Correct dataset citation in GAIC TASk chapter
--> Experimented with Few Shot prompting -> No relevant results

- Finish GAIC task chapter
- continue with Results and background

## KW 14

-> Gollie Finetuning with varying defintions 4
-> Start writing structuring of thesis
-> Prepare Marc Donnerstag

-> Start writing thesis try to interpret results first basic sections
-> Start exploring finetuning via Gollie

## KW 13

-> finishing data contamination experiments
-> Bringing it together with the other
-> Whats now the best idea for Improving the system

## KW 12

-> Our solution to marks problem is -> no training in RQ1 and 2
-> Whats the most clever way to train this system without shortcut
-> Gollie Fine Tuning

-> Data Contamination testing with the DCQ method by Golchin
-> Need to finish it seems promising

## KW 11

more details in [KW11](docs/diaries/KW11)

- Maybe for part 2 the comparison should be done especially for c2,c3 more explicitly among the 4 datasets -> Heads up comparison.

- "ERROR": {
  -> Many Executions contain an Third Error Class
- Alle 4.1 und 14b

- Scale up eperiments
  - yes, But first finish V1 Experiment and decide later

- Finetuning Results
  - Finetuning works but is flaky look deeper into best approach -> First finish Part 1,2
  - but generally possible!
-

- conducted Evaluation of how good the context is via [gaic/context_quality](gaic/context_quality_experiment.py)
  - next step would be to look at annotation guideline in isolation without definition
