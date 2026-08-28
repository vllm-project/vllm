{
  description = "Viper";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    devenv.url = "github:cachix/devenv";
  };

  outputs =
    inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [
        inputs.devenv.flakeModule
      ];

      systems = [
        "x86_64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];

      perSystem =
        { pkgs, ... }:
        {
          devenv.shells = {
            default = {
              languages = {
                go.enable = true;
              };

              git-hooks.hooks = {
                nixpkgs-fmt.enable = true;
                yamllint.enable = true;
              };

              packages = with pkgs; [
                gnumake

                golangci-lint
                yamllint
              ];

              scripts = {
                versions.exec = ''
                  go version
                  golangci-lint version
                '';
              };

              enterShell = ''
                versions
              '';

              # https://github.com/cachix/devenv/issues/528#issuecomment-1556108767
              containers = pkgs.lib.mkForce { };
            };
          };
        };
    };
}
