#!/usr/bin/env -S nix run .#
eval "$(micromamba shell hook -s bash)"
micromamba activate bmhrl
python runTraining.py --scorer 'METEOR' --one_by_one_starts_at 50
