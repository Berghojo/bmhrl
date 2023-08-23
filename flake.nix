{
  description = "Flakes-based nix shell for motilitAI based on micromamba";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  inputs.flake-utils.url = "github:numtide/flake-utils";

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    ...
  }:
    flake-utils.lib.eachDefaultSystem
    (
      system: let
        name = "micromamba-shell";

        pkgs = import nixpkgs {
          config = {
            # CUDA and other "friends" contain unfree licenses. To install them, you need this line:
            allowUnfree = true;
            permittedInsecurePackages = [
              "nodejs-16.20.0"
            ];
          };
          inherit system;
        };
        python-with-packages = pkgs.python3.withPackages (p:
          with p; [
            pip
            # other python packages you want
          ]);
      in rec {
        defaultPackage = pkgs.buildFHSUserEnv {
          inherit name;

          targetPkgs = pkgs:
            with pkgs; [
              micromamba
              libGL
              which
              libsndfile.out
              python-with-packages
              code-server
              zlib
            ];

          profile = ''
            set -e
            export MAMBA_ROOT_PREFIX=''${MAMBA_ROOT_PREFIX:-$HOME/micromamba}
            eval "$(micromamba shell hook -s posix)"
            set +e
          '';
          runScript = ''bash --init-file "/etc/profile"'';
        };
        packages = {
          ${name} = defaultPackage;
        };
        devShell = defaultPackage.env;
      }
    );
}
