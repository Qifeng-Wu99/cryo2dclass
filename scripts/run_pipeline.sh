#!/usr/bin/env bash
# Stage-by-stage cryo2dclass run. Edit the variables, then:  bash scripts/run_pipeline.sh
# For several GPUs replace `cryo2dclass train` with `accelerate launch -m cryo2dclass train`.
set -euo pipefail

DATA=/path/to/particles.mrcs   # .mrcs/.mrc/.npy stack, box divisible by 16
OUT=runs/exp1
K=2                            # number of classes to report
LABELS=""                      # optional ground truth, e.g. "--labels labels.txt"

cryo2dclass train --stage 1 --data "$DATA" --out "$OUT/stage1"
cryo2dclass train --stage 2 --data "$DATA" --out "$OUT/stage2" --init-from "$OUT/stage1"
cryo2dclass train --stage 3 --data "$DATA" --out "$OUT/stage3" --init-from "$OUT/stage2"

cryo2dclass evaluate  "$OUT/stage3" --n-clusters "$K" $LABELS
cryo2dclass visualize "$OUT/stage3"
