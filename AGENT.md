# mini-arc-v12 agent handoff

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

Select a profile with `--model-profile`; use the full wrapper
`scripts/train_mini_arc_v12_full_oar.sh` for Grid'5000. It defaults to
`$HOME/arc-checkpoints/mini-arc-v12-full`, and must never be pointed at the
reduced baseline checkpoint directory.

### Current full-model run

The full profile was trained on 8 A100s through epoch 42. Its latest observed
held-out RE-ARC metrics were 94.49% validation cell accuracy and 26.63% exact
grid accuracy. `latest.pt` is the resumable state; `best.pt` remains the state
with the lowest validation loss and is the checkpoint to evaluate first.

The epoch-42 stop was early stopping, not a training failure. The
`train_mini_arc_v12_full_sirius_night_resume.sh` wrapper raises patience to 100
and resumes the same run through epoch 100 without `--force-restart`.

### ARC-AGI evaluation and TTT

`arc_prize/eval_arc_agi.py` is the evaluator for the new checkpoint format; the
older Modal evaluator expects a different checkpoint schema and must not be
used directly. It performs direct inference and optional test-time training
(TTT): for each ARC task it copies the base model, adapts the copy using that
task's demonstrations, predicts its query, and discards the copy. TTT does not
change the base checkpoint or model architecture.

The current baseline never passes a noisy target through `tgt_embedding` during
pre-training. Consequently its target-refinement branch is untrained; do not
run a second `tgt=` refinement pass on this checkpoint. A faithful refinement
variant requires a new training profile and new checkpoints.

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

On Grid'5000, keep persistent checkpoints in:

```text
$HOME/arc-checkpoints/mini-arc-v12
```

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
```

`scripts/train_mini_arc_v12_oar.sh` is valid both as an OAR submission script
and when executed directly inside an interactive OAR reservation. It copies the
SIF, source, prepared dataset, and checkpoint into:

```text
/tmp/$USER/mini-arc-v12-$OAR_JOB_ID
```

It launches `torch.distributed.run` inside `apptainer exec --nv`, detects all
visible GPUs, and mirrors results back to `$HOME` after each checkpoint. Do not
change it to train directly from NFS-backed `$HOME`.

## Verification order

1. Run `python3 -m unittest discover -s tests -v` for preparation tests.
2. Build the SIF and run `apptainer exec --nv ... torch.cuda.device_count()` in
   the allocation.
3. Run the two-step smoke test documented in `TRAINING_MINI_ARC_V12.md`.
4. Check that both persistent `latest.pt` and `best.pt` exist.
5. Re-run with the same parameters, higher `MAX_EPOCHS`, and no forced restart
   to verify resume before a long allocation.

## Boundaries

- Do not commit generated datasets, SIF images, checkpoints, or `/tmp` files.
- Do not delete existing checkpoints; use the trainer's archive behavior.
- Keep the raw RE-ARC data outside the training source transfer when the
  prepared artifact is available.
- PyTorch-dependent tests require the Apptainer image or another environment
  with PyTorch. The manifest tests run with Python standard library only.
