# mini-arc-v12 agent handoff

## Immediate handoff state (2026-09-03, Grid'5000 local time)

The user is at Grid'5000 Lyon and wants to use the remaining Sirius night
window. The next compute tasks are:

1. Train the new `full-refinement` profile on Sirius with 8 A100s.
2. Evaluate the already-trained direct-only full checkpoint on a different,
   modern single-GPU node using the corrected paper-aligned evaluator.
3. After refinement training, evaluate its `best.pt` with TTT plus two
   refinement rounds.

The corrected Sirius reservation has **not been confirmed as submitted** in
this conversation. The earlier proposed interval from 19:05 to 08:55 was wrong
for a submission made around 23:30 because it represented 13h50. The wrapper
now explicitly contains `walltime=8:55:00`. For the night from September 3 to
September 4, submit exactly:

```bash
cd "$HOME/arc-agi/mini-arc"
oarsub \
  -r "2026-09-04 00:00:00,2026-09-04 08:55:00" \
  -S ./scripts/train_mini_arc_v12_full_refinement_sirius_night.sh
```

The wrapper includes `-t exotic`, `-t night`, host
`sirius-1.lyon.grid5000.fr`, 8 GPUs, and `walltime=8:55:00`. OAR may warn that
the reservation end is ignored because walltime is explicit; this is safe only
when `start_time=00:00:00` and `walltime=8:55:00`. Immediately verify with:

```bash
oarstat -fj <JOB_ID>
```

Do not assume success unless it shows:

```text
start_time = 2026-09-04 00:00:00
walltime = 8:55:00
types = exotic, night, ...
properties = (host='sirius-1.lyon.grid5000.fr') ...
```

If an incorrectly scheduled job exists, the user should identify it first and
cancel only that job with `oardel <WRONG_JOB_ID>`.

### Git synchronization blocker

The relevant local commits are:

```text
93290b6 Align ARC evaluation and add refinement training
beec1e6 Add Sirius night refinement launcher
22d9f10 Limit Sirius night refinement walltime
```

The Codex environment could commit but could not push over HTTPS because it had
no GitHub credentials (`could not read Username`). Before Grid'5000 can pull
these changes, the user must push from an authenticated local shell:

```bash
cd /home/eduardoaltmann/Workspace/arc-agi/mini-arc
git push origin main
```

Then on Grid'5000:

```bash
cd "$HOME/arc-agi/mini-arc"
git pull origin main
git log -4 --oneline
```

Verify that the three commits above and
`scripts/train_mini_arc_v12_full_refinement_sirius_night.sh` are present before
submitting. A later commit that updates this handoff may sit on top of them.

## Purpose

This repository contains an experimental UFRGS class-project training path for
`mini-arc-v12`: a reduced 12×12, 2×2 patch-based ARC vision encoder trained on
reduced RE-ARC. Preserve the pre-existing notebooks, Modal code, and older
training paths unless a task explicitly targets them.

## Data pipeline

Raw reduced RE-ARC data is a folder of generator-family JSON arrays:

```text
re-arc/re_arc_5k_12x12/tasks/*.json
```

Run the standard-library-only preparation command from `mini-arc`:

```bash
python3 -m arc_prize.rearc_manifest \
  --source ../re-arc/re_arc_5k_12x12 \
  --output data/re_arc_5k_12x12_balanced
```

The generated directory is intentionally ignored by Git and contains:

- `manifest.json`: deterministic source hashes, splits, exclusions, and
  dataset fingerprint;
- `examples.sqlite3`: compact, validated grid storage.

For the current source, expected preparation output is 391 eligible families,
186,556 retained examples, and fingerprint:

```text
f4c9714c940a05771462d1ee55c4f82fabbd6d56d3d7cd782299de8ce8522483
```

Families with fewer than six examples are excluded. Each validation target is
drawn from the held-out split; its four demonstrations are selected only from
the training split. `BalancedReARCDataset` must retain this property.

This is **only the reduced RE-ARC source**, not the full training mixture used
in the paper. The current project has 186,556 retained examples across 391
families. The paper reports roughly 830,648 training puzzles from RE-ARC, BARC
Heavy, and ARC-HTML and at least 150,000 optimization steps. Dataset diversity
is therefore the largest remaining reproduction gap. Do not claim that this
checkpoint reproduces the paper's result exactly.

## Training implementation

`arc_prize/train_rearc.py` is the authoritative v12 trainer.

- Model: `ARCVisionEncoder`, `grid_dim=12`, four demonstration pairs, 2×2
  patches, 4 layers, 4 heads, `d_model=128`, `d_ff=512`, dropout `0.1`.
- Default optimizer: AdamW, learning rate `1e-4`, weight decay `1e-4`.
- Default epoch: 1,000 train steps and 100 validation steps per DDP rank.
- Best checkpoint criterion: held-out RE-ARC validation loss.
- Reported metrics: training loss/cell accuracy and validation loss/cell/exact
  grid accuracy.

Architecture profiles are checkpoint-incompatible by design:

- `reduced` / `mini-arc-v12`: 4 layers, 4 heads, `d_model=128`, `d_ff=512`.
- `full` / `mini-arc-v12-full`: 16 layers, 16 heads, `d_model=512`, `d_ff=3072`.
- `full-refinement` / `mini-arc-v12-full-refinement`: the same full
  architecture (67,343,755 parameters), independently trained with noisy
  targets on 25% of steps.

Select a profile with `--model-profile`; use the full wrapper
`scripts/train_mini_arc_v12_full_oar.sh` for Grid'5000. It defaults to
`$HOME/arc-checkpoints/mini-arc-v12-full`, and must never be pointed at the
reduced baseline checkpoint directory.

### Current full-model run

The direct-only full profile completed 100,000 steps on 8 A100s. Its first ARC
evaluation covered 87 dimension-compatible tasks and scored zero exact grids;
two-epoch TTT increased padded cell accuracy from 82.83% to 88.68%. That run was
not paper-comparable because it excluded tasks with more than four context
pairs and used a shortened TTT procedure.

Observed persistent files were approximately 771 MiB each:

```text
$HOME/arc-checkpoints/mini-arc-v12-full/latest.pt
$HOME/arc-checkpoints/mini-arc-v12-full/best.pt
```

The reduced 4-layer infrastructure baseline also completed 100,000 steps. Its
last reported synthetic validation metrics were approximately 90.19% cell
accuracy, 4.34% exact-grid accuracy, and loss 0.4096. Its checkpoints were
approximately 9.6 MiB each under `$HOME/arc-checkpoints/mini-arc-v12`.

As of this handoff, no completed `full-refinement` result has been reported.
Its persistent directory is:

```text
$HOME/arc-checkpoints/mini-arc-v12-full-refinement
```

### ARC-AGI evaluation and TTT

`arc_prize/eval_arc_agi.py` is the evaluator for the new checkpoint format; the
older Modal evaluator expects a different checkpoint schema and must not be
used directly. It performs direct inference and optional test-time training
(TTT): for each ARC task it copies the base model, adapts the copy using that
task's demonstrations, predicts its query, and discards the copy. TTT does not
change the base checkpoint or model architecture. The paper-aligned defaults
are 15 epochs, a 99.5% cutoff, and all permutations of combinations with at
least three pairs. `data/mini_arc_v12_evaluation_ids.txt` supplies the paper's
114 IDs; the OAR launcher truncates context to the first four pairs and scores
the first query, matching the checked-in original experiment artifacts.

Metrics intentionally retain compatibility aliases. `score` is the count of
fully solved puzzles, `accuracy`/`cell_accuracy` includes all padded 12×12
cells, and `closeness` counts puzzles with at least 95% cell accuracy.

Cell accuracy is not puzzle accuracy. For example, a centered 4×4 output has
128 padding cells among 144 positions, so predicting only the padding correctly
already gives 88.9% cell accuracy while solving zero puzzles. In the first
evaluation, `exact_grid_accuracy=0.0` meant 0/89 query grids were fully correct.
In the paper, `TTT + Refined Score 20 (17.5%)` means 20/114 whole puzzles were
correct; that is the comparable success metric.

The corrected evaluator and launcher now:

- select the 114 task IDs listed in the paper;
- use the first four demonstrations when more are available, matching the
  checked-in original experiment artifacts;
- score only the first query per puzzle, again matching those artifacts;
- generate all permutations of every combination of at least three context
  pairs (6 TTT items for 3 pairs, 48 for 4 pairs);
- run up to 15 TTT epochs with a 99.5% adaptation-accuracy cutoff;
- report paper-style `score`, `accuracy`, and `closeness` as well as legacy
  aliases.

The paper-aligned result for the project's current checkpoint is still unknown.
Do not reuse the earlier 87-task JSON as the corrected result.

The direct-only baseline never passes a noisy target through `tgt_embedding` during
pre-training. Consequently its target-refinement branch is untrained; do not
run a second `tgt=` refinement pass on this checkpoint. The evaluator rejects
that misuse. Train `scripts/train_mini_arc_v12_full_refinement_oar.sh` for the
separate refinement checkpoint, then evaluate it with `REFINEMENT_ROUNDS=2`.

The refinement trainer uses the original model's separate `tgt_embedding`
input path. On 25% of optimization steps it receives a partially retained true
target mixed with random class values; on the other 75% it predicts from the
learned output query. This does not alter the full model architecture, but it
requires a separately trained checkpoint because `tgt_embedding` in the
direct-only checkpoint never received gradients.

Do not replace the full checkpoint with model weights alone. A resumable
checkpoint includes model, optimizer, AMP scaler, schedulers, epoch/step,
history, best metric, dataset fingerprint, and per-rank RNG state.

## Checkpoint locations and semantics

The trainer can use a local checkpoint directory plus an optional persistent
mirror:

```text
--checkpoint-dir <local path>
--persistent-checkpoint-dir <persistent path>
```

For every checkpoint, it atomically saves locally first, then atomically copies
`latest.pt` to the persistent directory. When validation improves, it also
copies `best.pt`. A normal start automatically resumes `latest.pt`; a forced
restart archives, rather than deletes, old checkpoints.

On Grid'5000, keep persistent checkpoints in separate directories:

```text
$HOME/arc-checkpoints/mini-arc-v12
$HOME/arc-checkpoints/mini-arc-v12-full
$HOME/arc-checkpoints/mini-arc-v12-full-refinement
```

`latest.pt` is the allocation-safe resume state. `best.pt` is selected by
lowest held-out RE-ARC validation loss and is the first checkpoint to evaluate.
Never copy weights alone: optimizer, scaler, schedulers, epoch/step, best loss,
history, dataset fingerprint, and per-rank RNG states are all stored.

## Apptainer and Grid'5000

Build the image defined by `containers/mini-arc-v12.def` with:

```bash
./scripts/build_mini_arc_v12_container.sh \
  ./mini-arc-v12-pytorch2.4.1-cuda12.1.sif
```

The expected persistent Grid'5000 layout is:

```text
$HOME/arc-agi/mini-arc
$HOME/arc-data/re_arc_5k_12x12_balanced
$HOME/containers/mini-arc-v12-pytorch2.4.1-cuda12.1.sif
$HOME/arc-checkpoints/mini-arc-v12
$HOME/arc-checkpoints/mini-arc-v12-full
$HOME/arc-checkpoints/mini-arc-v12-full-refinement
$HOME/arc-results
```

The last reported Grid'5000 persistent artifacts were:

```text
$HOME/arc-data/re_arc_5k_12x12_balanced/manifest.json
$HOME/arc-data/re_arc_5k_12x12_balanced/examples.sqlite3
$HOME/containers/mini-arc-v12-pytorch2.4.1-cuda12.1.sif
```

The SIF was approximately 3.0 GiB and successfully exposed PyTorch 2.4.1,
CUDA 12.1, and 8 A100 GPUs on Sirius.

`scripts/train_mini_arc_v12_oar.sh` is valid both as an OAR submission script
and when executed directly inside an interactive OAR reservation. It copies the
SIF, source, prepared dataset, and checkpoint into:

```text
/tmp/$USER/mini-arc-v12-$OAR_JOB_ID
```

It launches `torch.distributed.run` inside `apptainer exec --nv`, detects all
visible GPUs, and mirrors results back to `$HOME` after each checkpoint. Do not
change it to train directly from NFS-backed `$HOME`.

### Full-refinement training behavior

`scripts/train_mini_arc_v12_full_refinement_oar.sh` defaults to:

```text
MODEL_PROFILE=full-refinement
BATCH_SIZE=16 per GPU (global batch 128 on 8 GPUs)
TRAIN_STEPS=1000 per epoch
VALIDATION_STEPS=100 per epoch
MAX_EPOCHS=150 (150,000 steps)
PATIENCE=150
REFINEMENT_RATIO=0.25
CHECKPOINT_EVERY_STEPS=0
```

With `CHECKPOINT_EVERY_STEPS=0`, it still saves after every epoch. A benchmark
of the direct full model was about 128 seconds per 1,000-step epoch on 8 A100s,
so 150 epochs were estimated around 5–6 hours. The actual refinement duration
must be measured from its logs. It starts fresh if its directory has no
`latest.pt`; otherwise it resumes automatically. Do not set `FORCE_RESTART=1`
unless the user explicitly wants existing refinement checkpoints archived and
a new run.

### Evaluation commands

The user intends to evaluate the direct-only model on a different machine. In
an interactive allocation with one modern CUDA-capable GPU:

```bash
cd "$HOME/arc-agi/mini-arc"
RESULTS_DIR="$HOME/arc-results/mini-arc-v12-full-paper-ttt15" \
./scripts/eval_mini_arc_v12_full_oar.sh
```

This defaults to `$HOME/arc-checkpoints/mini-arc-v12-full/best.pt`, TTT=15, and
no refinement. It uses only `cuda:0`. A prior five-task/two-epoch smoke took
about 30 seconds; the 114-task/15-epoch evaluation was conservatively estimated
at 2–3 hours, but no final timing has been reported. The evaluation itself is
not step-checkpointed, so leave enough walltime.

After the refinement run finishes:

```bash
cd "$HOME/arc-agi/mini-arc"
CHECKPOINT_PATH="$HOME/arc-checkpoints/mini-arc-v12-full-refinement/best.pt" \
RESULTS_DIR="$HOME/arc-results/mini-arc-v12-full-refinement-paper" \
REFINEMENT_ROUNDS=2 \
./scripts/eval_mini_arc_v12_full_oar.sh
```

Inspect either result with:

```bash
jq '{
  tasks_requested,
  tasks_eligible,
  skipped_tasks: (.tasks_skipped | length),
  direct_metrics,
  ttt_metrics,
  refined_metrics
}' "$HOME/arc-results/<RUN>/arc-agi-evaluation.json"
```

`refined_metrics` is present only when `REFINEMENT_ROUNDS` is greater than
zero. Do not use a Kepler-era K20M with the CUDA 12.1 / PyTorch 2.4.1 image;
use an A100, L40S, H200, V100 if compatible with the image, or another supported
modern NVIDIA GPU.

### Logs and interruption recovery

OAR logs are written in the repository launch directory using `%jobid%`, for
example:

```text
mini-arc-v12-full-refinement-<jobid>.out
mini-arc-v12-full-refinement-<jobid>.err
```

Useful checks are:

```bash
oarstat -fj <JOB_ID>
tail -n 80 mini-arc-v12-full-refinement-<JOB_ID>.out
tail -n 80 mini-arc-v12-full-refinement-<JOB_ID>.err
ls -lh "$HOME/arc-checkpoints/mini-arc-v12-full-refinement"
```

The generic launcher stages source, SIF, prepared data, and checkpoints under
`/tmp/$USER/mini-arc-v12-$OAR_JOB_ID`. On normal completion it mirrors results
to `$HOME` and removes only that job-specific scratch directory. On termination
it forwards SIGTERM to the trainer, which saves at the next batch boundary and
syncs. Even if the final emergency save fails, the most recently completed
epoch checkpoint remains persistent.

## Verification and tests already completed

The following passed inside the Apptainer image after the latest evaluator and
refinement changes:

- all five repository unit tests;
- Python compilation and Bash syntax checks;
- TTT construction counts of 6 items for three pairs and 48 for four pairs;
- Score/Accuracy/Closeness aggregation;
- a small model forward/backward through `tgt_embedding`, confirming it receives
  gradients.

On the actual cluster, still verify each new job's startup JSON, visible GPU
count, model name, parameter count, checkpoint directory, and final persistent
files. The refinement startup should report model name
`mini-arc-v12-full-refinement`, parameter count 67,343,755, world size 8, and
`refinement_ratio: 0.25`.

## Boundaries

- Do not commit generated datasets, SIF images, checkpoints, or `/tmp` files.
- Do not delete existing checkpoints; use the trainer's archive behavior.
- Keep the raw RE-ARC data outside the training source transfer when the
  prepared artifact is available.
- PyTorch-dependent tests require the Apptainer image or another environment
  with PyTorch. The manifest tests run with Python standard library only.
- Never run refinement on `mini-arc-v12-full/best.pt`; that checkpoint's
  `tgt_embedding` is untrained and the evaluator intentionally rejects it.
- Keep the 87-task old evaluation for historical reference, but use the new
  114-task output for paper comparisons.
- Grid'5000 usage-policy safety is a hard requirement: use `-t night`, inspect
  `start_time` and `walltime`, and end before 09:00 local site time.
