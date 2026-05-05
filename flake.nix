{
  description = "Ambiente para desenvolvimento de atividades do curso SIN5007";
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };
  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
    in
      {
        devShells.${system}.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            git
            uv
            python315
          ];

          shellHook = ''
                    echo "Shell para SIN5007 inicializada..."
                    '';
        };
      };

}
