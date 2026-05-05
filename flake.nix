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

          nativeBuildInputs = with pkgs; [
            pkg-config
          ];
          buildInputs = with pkgs; [
            git
            uv
            python312
            ninja
            zlib
            libjpeg
            freetype
          ];

          shellHook = ''
                    export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath (with pkgs; [
                   zlib
                   libjpeg
                   freetype
                   stdenv.cc.cc.lib
                 ])}:$LD_LIBRARY_PATH
                    echo "Shell para SIN5007 inicializada..."
                    '';
        };
      };

}
