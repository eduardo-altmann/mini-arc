#!/usr/bin/env bash
#OAR -n mini-arc-v12-full-refinement
#OAR -q default
#OAR -t exotic
#OAR -t night
#OAR -p host='sirius-1.lyon.grid5000.fr'
#OAR -l host=1/gpu=8,walltime=8:55:00
#OAR -O mini-arc-v12-full-refinement-%jobid%.out
#OAR -E mini-arc-v12-full-refinement-%jobid%.err

# Submit with oarsub -r to make the reservation end before restricted daytime.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$script_dir/train_mini_arc_v12_full_refinement_oar.sh"
