#!/usr/bin/env -S nix run .#
eval "$(micromamba shell hook -s bash)"
micromamba activate bmhrl
python runTraining_Meteor.py