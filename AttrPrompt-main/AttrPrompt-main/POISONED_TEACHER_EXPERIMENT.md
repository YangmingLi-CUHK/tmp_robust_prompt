# End-to-end poisoned-teacher AttrPrompt

This is a poisoning experiment, not a clean-pretrained-teacher experiment.

For every Metattack rate `r`, the pipeline is:

```text
A_M-r + train labels
    -> train and select GCN teacher on A_M-r
    -> freeze this M-r teacher
    -> reference = teacher_M-r(X, A_M-r)
    -> student   = teacher_M-r((X + prompt)/2, A'_M-r)
    -> optimize prompt by KL [+ IB]
    -> test teacher and prompt on A_M-r
```

`A'_M-r` is AttrPrompt's internal perturbation around `A_M-r`. For every
non-zero rate, neither Phase 1 nor Phase 2 loads the clean adjacency tensor.
Each run loads its matching `M-r` train/validation/test index files. Node
attributes remain unchanged by this structural attack.

Each teacher checkpoint has a JSON sidecar containing:

- teacher training rate;
- graph source filename;
- SHA-256 graph-support fingerprint;
- SHA-256 node-split fingerprint;
- whether the clean adjacency was loaded;
- validation-selected test accuracy.

Phase 2 checks the rate and fingerprint before loading the teacher. A clean
teacher or a teacher from another rate causes the run to stop.

## Server commands

Run all rates:

```bash
cd /home/tony/LnL/DFS_HK5/AttrPrompt-main/AttrPrompt-main
conda activate LnL2
bash run_poisoned_teacher_attrprompt_server.sh
```

Smoke-test only M-0.25:

```bash
PTB_RATES="0.25" SEEDS=1 TEACHER_EPOCHS=5 PROMPT_EPOCHS=5 \
OUTPUT_ROOT=./save_cora/poisoned_teacher_smoke \
bash run_poisoned_teacher_attrprompt_server.sh
```

Resume prompt experiments from already-generated poisoned teachers:

```bash
RETRAIN_TEACHER=0 bash run_poisoned_teacher_attrprompt_server.sh
```

Outputs are separated by rate:

```text
save_cora/poisoned_teacher_attrprompt/
  M0p0/
    GCN/
    AttrPrompt_dynamic/
  M0p05/
  ...
  M0p25/
  poisoned_pipeline_summary.csv
```

The combined CSV reports the supervised poisoned-teacher accuracy, frozen
teacher direct-forward accuracy during Phase 2, AttrPrompt accuracy, prompt
gain, F1, and structure-sanity measurements.
