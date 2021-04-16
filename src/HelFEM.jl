module HelFEM
using ReadOnlyArrays
using CxxWrap: CxxPtr

export PolynomialBasis, RadialBasis, FEMBasis,
    boundaries, nelements, radial_integral

include("helfem_cxxwrap.jl")
include("polynomialbasis.jl")
include("radialbasis.jl")
include("fembasis.jl")

invh(S) = helfem.invh(S, false)

## Quadrature

"""
    chebyshev(nquad) -> (xs::Vector, ws::Vector)

Return HelFEM's modified Gauss-Chebyshev quadrature points and weights for numerical
integration of functions on ``x \\in [-1, 1]``.
"""
function chebyshev(nquad)
    xs, ws = HelFEM.helfem.ArmaVector(), HelFEM.helfem.ArmaVector()
    HelFEM.helfem.chebyshev(nquad, xs, ws)
    (xs = collect(xs), ws = collect(ws))
end

"""
    lobatto(nquad) -> (xs::Vector, ws::Vector)

Return HelFEM's Gauss-Lobatto quadrature points and weights for numerical on
``x \\in [-1, 1]``.
"""
function lobatto(nquad)
    xs, ws = HelFEM.helfem.ArmaVector(), HelFEM.helfem.ArmaVector()
    HelFEM.helfem.lobatto(nquad, xs, ws)
    (xs = collect(xs), ws = collect(ws))
end

end # module
