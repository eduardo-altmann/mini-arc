# Training mini-arc-v12 on reduced RE-ARC

This path trains a fresh 12×12 vision encoder with 2×2 patches. It is separate
from the older notebook and Modal workflows.

## 1. Prepare and inspect the dataset

From the `mini-arc` directory:

```bash
python -m arc_prize.rearc_manifest \
  --source ../re-arc/re_arc_5k_12x12 \
  --output "$HOME/arc-data/re_arc_5k_12x12_balanced"
```

Preparation validates every grid and creates `manifest.json` plus a compact
read-only `examples.sqlite3`. With the current data it retains 391 families and
186,556 examples, excluding three families that have only two examples. Do not
use `--overwrite` unless intentionally rebuilding the prepared dataset.

## 2. CPU/single-process smoke test

Use very small step counts first. `--force-restart` archives existing `latest.pt`
and `best.pt` before starting a fresh run.

```bash
python -m arc_prize.train_rearc \
  --dataset-dir "$HOME/arc-data/re_arc_5k_12x12_balanced" \
  --checkpoint-dir "$HOME/arc-checkpoints/mini-arc-v12-smoke" \
  --batch-size 2 --train-steps 2 --validation-steps 1 --max-epochs 1 \
  --force-restart
```

PyTorch automatically uses CUDA when visible and CPU otherwise. A normal launch
automatically resumes `latest.pt`. Configuration and dataset fingerprint checks
prevent accidental resume with incompatible data or hyperparameters.

## 3. Grid'5000/OAR run

Copy the workspace to `$HOME/arc-agi`, create/activate an environment containing
the dependencies from `pyproject.toml`, and submit from a persistent directory:

```bash
cd "$HOME/arc-agi/mini-arc"
oarsub -S ./scripts/train_mini_arc_v12_oar.sh
```

The script defaults to `host=1/gpu=2,walltime=4:00:00`. Override the OAR resource
request at submission when needed, for example:

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

The job stages code and prepared data in
`/tmp/$USER/mini-arc-v12-$OAR_JOB_ID`. Persistent files are:

- `$HOME/arc-checkpoints/mini-arc-v12/latest.pt`: current model and complete
  optimizer, AMP, scheduler, sampler progress, metrics, and per-rank RNG state.
- `$HOME/arc-checkpoints/mini-arc-v12/best.pt`: the complete state from the
  lowest held-out RE-ARC validation loss.
- `mini-arc-v12-<jobid>.out` and `.err`: OAR logs in the submission directory.

On `SIGTERM` or `SIGINT`, all ranks checkpoint at the next completed batch. If a
full epoch is comfortably shorter than the allocation, leave
`CHECKPOINT_EVERY_STEPS=0`. Otherwise set a fixed interval such as `200`.

## 4. Progressive run sequence

1. Run the CPU/single-process smoke command above.
2. Submit one short multi-GPU epoch with small step counts.
3. Submit the same command again without `--force-restart` and confirm the log
   reports the next epoch/global step.
4. Benchmark one full 1,000-step epoch and check GPU memory and duration.
5. Continue normal allocations; every new job resumes `latest.pt` automatically.
