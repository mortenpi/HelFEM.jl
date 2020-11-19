# HelFEM

<!-- [![][docs-stable-img]][docs-stable-url] -->
[![][docs-dev-img]][docs-dev-url]

Provides a Julia wrapper for the [HelFEM finite element method library](https://github.com/susilehtola/HelFEM).
Uses [HelFEM_jll](https://github.com/JuliaPackaging/Yggdrasil/blob/master/H/HelFEM/build_tarballs.jl) ([JuliaHub](https://juliahub.com/ui/Packages/HelFEM_jll/Tv3dF/0.0.1+1)) which provides compiled binaries of the [CxxWrap](https://github.com/JuliaInterop/CxxWrap.jl)-based wrapper.

The package is not yet registered, but can be installed in the Julia Pkg REPL-mode (accessible via `]`) by the URL:

```julia-repl
pkg> add https://github.com/mortenpi/HelFEM.jl.git
```

[docs-dev-img]: https://img.shields.io/badge/docs-development-blue.svg
[docs-dev-url]: http://mortenpi.eu/HelFEM.jl/dev/
[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: http://mortenpi.eu/HelFEM.jl/stable/
