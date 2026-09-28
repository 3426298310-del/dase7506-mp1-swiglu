# Pre-test experiment plan

Created before any test-set scoring in this work session.

## Question
Does a near-parameter-matched SwiGLU feed-forward sublayer improve validation BPB
over the supplied GELU GPT at the same number of processed training targets?
This is an application and controlled evaluation of an existing mechanism, not
a claim to have invented SwiGLU (Shazeer, 2020, https://arxiv.org/abs/2002.05202).

## Controls
Keep data, tokenizer, independent causal context=256, model width/depth/heads,
AdamW, warmup, cosine schedule, batch size, and training precision fixed in each
paired experiment. Compare final checkpoints at 1,200 steps / 9,830,400 targets.
Repeat the pair with seeds 17 and 29 if runtime permits. The gated=False branch
is tested to be exactly the original baseline and serves as the mechanism ablation.
The SwiGLU inner width is 344, versus 512 for the baseline GELU network.

## Selection and possible extension
Only full validation BPB is used for selection. Choose between the two mechanisms
using paired results, then investigate a longer training budget separately if
validation curves justify it. Record all attempted runs, including failures.
Select the best validation checkpoint of the candidate runs; account for all
completed training updates, including those after the selected checkpoint.
No cached answers, external training corpus, pretrained weights, or test tuning.

## Freeze and final audit
Freeze the chosen checkpoint and source hashes in results/freeze.json before
the first full-test score. Run the fixed FP32 CPU scorer. Benchmark the baseline
and frozen candidate in fresh processes, alternating order, using four CPU threads.
Report median scoring time of three trials and maximum measured process peak RAM.
Windows peak working set is sampled through psutil; report this measurement method.
Count the uncompressed checkpoint and all extra inference assets against 64 MiB.
Preserve the original evaluator, common.py, data, tokenizer and baseline.

## Disclosure
Codex assisted with planning, implementation, experiment execution, analysis and
documentation. The student must review and understand this work before submission.
No scores are claimed until actual completed evaluator outputs exist.
