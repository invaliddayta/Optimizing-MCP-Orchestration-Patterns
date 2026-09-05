{
  description = "A local MCP orchestration benchmark, with a Nix-only toolchain";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" ];
      eachSystem = nixpkgs.lib.genAttrs systems;
      environment = system:
        let
          pkgs = import nixpkgs { inherit system; };
          python = pkgs.python3.withPackages (ps: [ ps.aiohttp ps.mcp ps.jsonschema ]);
          source = pkgs.lib.fileset.toSource {
            root = ./.;
            fileset = pkgs.lib.fileset.unions [
              (pkgs.lib.fileset.fileFilter (file: file.hasExt "py") ./lab)
              (pkgs.lib.fileset.fileFilter (file: file.hasExt "py") ./tests)
              (pkgs.lib.fileset.fileFilter (file: file.hasExt "py") ./mcpservers)
              (pkgs.lib.fileset.fileFilter (file: file.hasExt "json") ./config)
              ./flake.lock ./flake.nix ./pyproject.toml
            ];
          };
          bench = pkgs.writeShellApplication {
            name = "mcp-bench";
            text = ''
              export PYTHONNOUSERSITE=1
              export PYTHONPATH=${source}
              exec ${python}/bin/python -P -m lab "$@"
            '';
          };
        in { inherit pkgs python source bench; };
    in {
      packages = eachSystem (system: let e = environment system; in {
        default = e.bench;
        python = e.python;
      });
      apps = eachSystem (system: {
        default = {
          type = "app";
          program = "${self.packages.${system}.default}/bin/mcp-bench";
          meta.description = "Run and inspect local MCP orchestration experiments";
        };
      });
      devShells = eachSystem (system: let e = environment system; in {
        default = e.pkgs.mkShell {
          packages = [ e.python e.pkgs.llama-cpp e.pkgs.ruff ];
          PYTHONNOUSERSITE = "1";
        };
      });
      checks = eachSystem (system: let e = environment system; in {
        tests = e.pkgs.runCommand "mcp-bench-tests" { nativeBuildInputs = [ e.python ]; } ''
          export PYTHONNOUSERSITE=1
          export PYTHONDONTWRITEBYTECODE=1
          export PYTHONPATH=${e.source}
          export MCP_BENCH_APP=${e.bench}/bin/mcp-bench
          python -m unittest discover -s ${e.source}/tests -v
          touch "$out"
        '';
        lint = e.pkgs.runCommand "mcp-bench-lint" { nativeBuildInputs = [ e.pkgs.ruff ]; } ''
          ruff check --config ${e.source}/pyproject.toml ${e.source}/lab ${e.source}/tests ${e.source}/mcpservers/calculator_mcp.py
          ruff format --check --config ${e.source}/pyproject.toml ${e.source}/lab ${e.source}/tests ${e.source}/mcpservers/calculator_mcp.py
          touch "$out"
        '';
      });
    };
}
