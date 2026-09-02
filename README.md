# mini-arc-v12 — UFRGS class project

## Credits

This repository is based on the original **mini-arc / ARC Prize** project created
by **Paul Fletcher-Hill**. The original project and paper are available at
[mini-arc.pdf](https://www.paulfletcherhill.com/mini-arc.pdf).

The `mini-arc-v12` training path is work for a class project at the
**Federal University of Rio Grande do Sul (UFRGS)**. It adds a reduced 12×12
patch-based model, balanced RE-ARC preparation, portable Apptainer execution,
and restart-safe training on Grid'5000.

## What this project trains

`mini-arc-v12` is a fresh `vision_encoder` model with:

- 12×12 input/output grids;
- four ARC demonstration pairs plus one query grid;
- 2×2 patch embeddings;
- four encoder layers, four attention heads, `d_model=128`, and `d_ff=512`.

The default `reduced` profile is the validated baseline. The separate `full`
profile reproduces the creator's architecture: 16 layers, 16 heads,
`d_model=512`, and `d_ff=3072`. It uses independent checkpoints and is launched
with `scripts/train_mini_arc_v12_full_oar.sh`.

Training uses the reduced RE-ARC dataset. Examples are sampled uniformly by
generator family, so large families do not dominate the objective. Validation
targets are held out per family and never appear as demonstrations.

## Project files

- `arc_prize/rearc_manifest.py` validates raw RE-ARC JSON and produces the
  prepared dataset.
- `arc_prize/rearc_dataset.py` creates deterministic, balanced training and
  validation tasks.
- `arc_prize/train_rearc.py` implements DDP training, metrics, checkpointing,
  signal handling, and resume.
- `containers/mini-arc-v12.def` defines the PyTorch/Apptainer image.
- `scripts/train_mini_arc_v12_oar.sh` stages a Grid'5000 job into `/tmp` and
  mirrors checkpoints back to `$HOME`.
- `scripts/train_mini_arc_v12_full_oar.sh` selects the original full profile
  and `$HOME/arc-checkpoints/mini-arc-v12-full`.

## Quick workflow

1. Prepare the dataset without a GPU:

   ```bash
   python3 -m arc_prize.rearc_manifest \
     --source ../re-arc/re_arc_5k_12x12 \
     --output data/re_arc_5k_12x12_balanced
   ```

2. Build the container on a Linux/amd64 machine with Apptainer:

   ```bash
   ./scripts/build_mini_arc_v12_container.sh \
     ./mini-arc-v12-pytorch2.4.1-cuda12.1.sif
   ```

3. Copy the prepared dataset, SIF, and source code to persistent Grid'5000
   `$HOME` storage.

4. Inside an OAR GPU allocation, verify the image with `apptainer exec --nv`,
   run a two-step smoke test, then start the launcher.

See [TRAINING_MINI_ARC_V12.md](TRAINING_MINI_ARC_V12.md) for exact transfer,
smoke-test, OAR, checkpoint, and resume commands. See [AGENT.md](AGENT.md) for
an implementation-oriented handoff.
