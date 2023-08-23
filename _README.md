# **conda shell**
This shell can be used for a traditional conda-based workflow - however, we replace conda with micromamba, as it is faster and can also be used outside of the shell for creating environments and installing dependencies.
## Usage
Create your conda environments as usual and install software, but replace `conda` with `micromamba`. E.g.
```bash
micromamba create -n MyCondaEnv
micromamba install -n MyCondaEnv scikit-learn -c conda-forge
```
Run either `nix run` or `nix develop` from this folder, to enter a FHS environment where micromamba works as usual.


## Scripts and SLURM Jobs
To activate your shell inside your scripts, add a shebang as in `./scripts.sh`.

A sample slurm script can be found in `gpujob.sh`.

