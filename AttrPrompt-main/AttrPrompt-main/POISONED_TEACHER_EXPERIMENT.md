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
RUN_ID=full_v1 bash run_poisoned_teacher_attrprompt_server.sh
```

Smoke-test only M-0.25:

```bash
PTB_RATES="0.25" SEEDS=1 TEACHER_EPOCHS=5 PROMPT_EPOCHS=5 \
OUTPUT_ROOT=./save_cora/poisoned_teacher_smoke \
RUN_ID=smoke_m025 \
bash run_poisoned_teacher_attrprompt_server.sh
```

Resume an interrupted run with the same rate set:

```bash
RUN_ID=full_v1 RESUME=1 bash run_poisoned_teacher_attrprompt_server.sh
```

Completed rates are detected from their rate-specific CSV and skipped. A
different rate set or an already-complete run is rejected.

Rates are normalized to two decimal places to match the dataset filenames.
For example, `0.1` and `0.10` both mean `0.10`; supplying both is rejected as
a duplicate before training starts.

Every invocation has an isolated run directory. Every rate also gets its own
CSV immediately after that rate finishes:

```text
save_cora/poisoned_teacher_attrprompt/
  runs/
    full_v1/
      M0p00/
        GCN/
        AttrPrompt_dynamic/
        result_M0p00.csv
      M0p05/
        result_M0p05.csv
      ...
      M0p25/
        result_M0p25.csv
      poisoned_pipeline_summary.csv
```

The script refuses to reuse an existing `RUN_ID` unless `RESUME=1` is supplied.
The combined CSV is written once, after all requested rates finish. It reports
the supervised poisoned-teacher accuracy, frozen teacher direct-forward
accuracy during Phase 2, AttrPrompt accuracy, prompt gain, F1, and
structure-sanity measurements.
