# Training mini-arc-v12 on reduced RE-ARC

This path trains a fresh 12×12 vision encoder with 2×2 patches. It is separate
from the older notebook and Modal workflows.

## Before the GPU reservation

Dataset preparation and container construction do not need a GPU. Do both on
another Linux machine when possible, then transfer the artifacts to the
Grid'5000 site's persistent `$HOME`.

### 1. Prepare and inspect the dataset

From the `mini-arc` directory:

```bash
python3 -m arc_prize.rearc_manifest \
  --source ../re-arc/re_arc_5k_12x12 \
  --output ./data/re_arc_5k_12x12_balanced
```

Preparation validates every grid and creates `manifest.json` plus a compact
read-only `examples.sqlite3`. With the current data it retains 391 families and
186,556 examples, excluding three families that have only two examples. It uses
only the Python standard library. The prepared directory is portable: its
fingerprint is based on content and split indices, not its filesystem path.

Copy both generated files to the Grid'5000 site:

```bash
rsync -av data/re_arc_5k_12x12_balanced/ \
  '<login>@access.grid5000.fr:<site>/arc-data/re_arc_5k_12x12_balanced/'
```

Do not use `--overwrite` unless intentionally rebuilding the prepared dataset.

### 2. Build the Apptainer image

On a Linux/amd64 machine with Apptainer and internet access:

```bash
cd mini-arc
./scripts/build_mini_arc_v12_container.sh \
  ./mini-arc-v12-pytorch2.4.1-cuda12.1.sif
```

The definition uses `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime`, matching the
project's PyTorch requirement while supporting a broader range of host driver
versions than CUDA 12.4. Copy the immutable SIF to the Grid'5000 site:

```bash
rsync -av --info=progress2 mini-arc-v12-pytorch2.4.1-cuda12.1.sif \
  '<login>@access.grid5000.fr:<site>/containers/mini-arc-v12-pytorch2.4.1-cuda12.1.sif'
```

If the other machine cannot build Apptainer images, run the build helper at the
start of the interactive reservation. Grid'5000's standard environment provides
Apptainer without requiring root, but this fallback must download a multi-GB
base image during the allocation.

### 3. Copy the source tree

Place the repository at `$HOME/arc-agi/mini-arc` on the same Grid'5000 site.
Only the prepared dataset is needed for training; copying the raw RE-ARC JSON is
optional once preparation has succeeded.

Create the `arc-data`, `containers`, and `arc-agi` destination directories on
the site frontend before transferring. Exclude both local RE-ARC data folders
when copying the source tree because the prepared artifact is transferred
separately.

## Inside tomorrow's interactive allocation

### 1. Verify the eight GPUs inside the container

```bash
apptainer exec --nv \
  "$HOME/containers/mini-arc-v12-pytorch2.4.1-cuda12.1.sif" \
  python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count()); assert torch.cuda.device_count() == 8'
```

### 2. Run a two-step, eight-GPU smoke test

The same launcher works when executed directly inside an interactive OAR job;
its `#OAR` lines are simply comments in that case.

```bash
cd "$HOME/arc-agi/mini-arc"
CHECKPOINT_DIR="$HOME/arc-checkpoints/mini-arc-v12-smoke" \
BATCH_SIZE=2 TRAIN_STEPS=2 VALIDATION_STEPS=1 MAX_EPOCHS=1 \
FORCE_RESTART=1 ./scripts/train_mini_arc_v12_oar.sh
```

Confirm that `latest.pt` and `best.pt` exist in the smoke checkpoint directory.
The launcher detects all eight visible A100s and starts eight DDP ranks.

### 3. Start or resume the real run

```bash
cd "$HOME/arc-agi/mini-arc"
./scripts/train_mini_arc_v12_oar.sh
```

For a later non-interactive allocation, submit the same launcher with an OAR
resource override:

```bash
oarsub -l host=1/gpu=4,walltime=8:00:00 -S ./scripts/train_mini_arc_v12_oar.sh
```

Runtime settings are environment variables:

```bash
BATCH_SIZE=16 TRAIN_STEPS=10 VALIDATION_STEPS=2 MAX_EPOCHS=1 \
  oarsub -S ./scripts/train_mini_arc_v12_oar.sh
```

The launcher also accepts `LEARNING_RATE`, `WEIGHT_DECAY`, `SEED`,
`NUM_WORKERS`, `PATIENCE`, `CHECKPOINT_EVERY_STEPS`, and `FORCE_RESTART=1`.
Use `FORCE_RESTART` carefully: it starts a new model but first archives the
existing `latest.pt` and `best.pt` instead of deleting them.

The job stages the SIF, source, prepared dataset, and any existing resumable
checkpoint under `/tmp/$USER/mini-arc-v12-$OAR_JOB_ID`. Training reads and writes
the local disk. Every completed checkpoint is then copied atomically to:

- `$HOME/arc-checkpoints/mini-arc-v12/latest.pt`: current model and complete
  optimizer, AMP, scheduler, sampler progress, metrics, and per-rank RNG state.
- `$HOME/arc-checkpoints/mini-arc-v12/best.pt`: the complete state from the
  lowest held-out RE-ARC validation loss.
- `mini-arc-v12-<jobid>.out` and `.err`: OAR logs in the submission directory.

On `SIGTERM` or `SIGINT`, all ranks checkpoint at the next completed batch and
the launcher synchronizes local results to `$HOME`. On normal completion it
performs a final synchronization and removes only its job-specific `/tmp`
directory. If a full epoch is comfortably shorter than the allocation, leave
`CHECKPOINT_EVERY_STEPS=0`. Otherwise set a fixed interval such as `200`.

## 4. Progressive run sequence

1. Run the two-step container smoke command above.
2. Run one short multi-GPU epoch with realistic batch size.
3. Run the same command again without `FORCE_RESTART` and confirm the log
   reports the next epoch/global step.
4. Benchmark one full 1,000-step epoch and check GPU memory and duration.
5. Continue normal allocations; every new job resumes `latest.pt` automatically.

## 5. Original mini-arc-v12 architecture

The reduced profile is a fast infrastructure baseline. The original architecture
is available separately through `scripts/train_mini_arc_v12_full_oar.sh`:

- 16 encoder layers, 16 heads, `d_model=512`, and `d_ff=3072`;
- 12×12 grids, four demonstration pairs, 2×2 patches, sequence length 468;
- persistent checkpoints in `$HOME/arc-checkpoints/mini-arc-v12-full`.

It deliberately cannot resume a reduced checkpoint. First run a short full-model
smoke test to establish memory use:

```bash
cd "$HOME/arc-agi/mini-arc"
CHECKPOINT_DIR="$HOME/arc-checkpoints/mini-arc-v12-full-smoke" \
BATCH_SIZE=2 TRAIN_STEPS=2 VALIDATION_STEPS=1 MAX_EPOCHS=1 \
FORCE_RESTART=1 ./scripts/train_mini_arc_v12_full_oar.sh
```

The startup JSON prints `model_name`, the model architecture, and
`parameter_count`. It should report `mini-arc-v12-full` and approximately
67.3 million parameters. Then benchmark one full epoch from a fresh full-model
checkpoint directory, increasing `BATCH_SIZE` only after confirming the A100
memory headroom.
