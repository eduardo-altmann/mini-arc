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
- `full-refinement` / `mini-arc-v12-full-refinement`: the same full
  architecture, independently trained with noisy targets on 25% of steps.

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

The direct-only baseline never passes a noisy target through `tgt_embedding` during
pre-training. Consequently its target-refinement branch is untrained; do not
run a second `tgt=` refinement pass on this checkpoint. The evaluator rejects
that misuse. Train `scripts/train_mini_arc_v12_full_refinement_oar.sh` for the
separate refinement checkpoint, then evaluate it with `REFINEMENT_ROUNDS=2`.

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
