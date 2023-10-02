#!/usr/bin/env -S nix run .#
eval "$(micromamba shell hook -s bash)"
micromamba activate bmhrl
python runTraining.py  --scorer='CIDER' --one_by_one_starts_at=50 --B=16 --rl_pretrained_model_dir="./checkpoints/364"