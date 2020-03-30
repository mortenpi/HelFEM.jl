module HelFEM
using CxxWrap: CxxPtr

module helfem
using CxxWrap
    #import armadillo_jll, OpenBLAS_jll, HDF5_jll
    @wrapmodule(joinpath(@__DIR__, "..", "deps", "lib", "libhelfem.so"))
    function __init__()
        @initcxx
        verbose(false)
    end

    Base.size(v::ArmaVector) = (Int(nrows(v)),)
    function Base.size(v::ArmaVector, dim::Integer)
        dim <= 0 && throw(ArgumentError("dimension $dim out of range"))
        dim == 1 ? Int(nrows(v)) : 1
    end
    Base.collect(v::ArmaVector) = [at(v, i - 1) for i = 1:size(v, 1)]

    Base.size(m::ArmaMatrix) = (Int(nrows(m)), Int(ncols(m)))
    function Base.size(m::ArmaMatrix, dim::Integer)
        dim <= 0 && throw(ArgumentError("dimension $dim out of range"))
        dim == 1 ? Int(nrows(m)) : dim == 2 ? Int(ncols(m)) : 1
    end
    Base.collect(m::ArmaMatrix) = [at(m, i - 1, j - 1) for i = 1:size(m, 1), j = 1:size(m, 2)]
end

struct RadialBasis
    b :: helfem.RadialBasis

    function RadialBasis(nnodes, nelem; primbas = 4, rmax = 40.0, igrid = 2, zexp = 2.0, nquad = nothing)
        new(helfem.basis(nnodes, nelem, primbas, rmax, igrid, zexp, isnothing(nquad) ? 0 : nquad))
    end
end

Base.length(b::RadialBasis) = Int(helfem.nbf(b.b))

function Base.show(io::IO, b::RadialBasis)
    nbf = helfem.nbf(b.b)
    nel = helfem.nel(b.b)
    write(io, "RadialBasis: $nbf basis functions ($nel elements)")
end

function overlap(b::RadialBasis; invh=false)
    S = helfem.overlap(b.b, b.b)
    if invh
        Sinvh = helfem.form_Sinvh(S, false)
        collect(S), collect(Sinvh)
    else
        collect(S)
    end
end

function radial_integral(b1::RadialBasis, n::Integer, b2::RadialBasis = b1; lderivative=false, rderivative=false)
    I = helfem.radial_integral(b1.b, b2.b, n, lderivative, rderivative)
    collect(I)
end

function potential(b1::RadialBasis, model::Symbol, rms::Real = 0.0, b2::RadialBasis=b1)
    if model === :point && rms != 0.0
        @warn "Point nuclear model should pass rms = 0.0. RMS value ignored." rms
    elseif model !== :point && rms <= 0.0
        throw(ArgumentError("Must pass rms >= 0 for nuclear model $model (passed $rms)"))
    end
    modelobj = (model === :point) ? helfem.PointNucleus(1) :
        (model === :gaussian)     ? helfem.GaussianNucleus(1, rms) :
        (model === :spherical)    ? helfem.SphericalNucleus(1, rms) :
        (model === :hollow)       ? helfem.HollowNucleus(1, rms) :
        throw(ArgumentError("Invalid nucler model: $(model)"))
    I = helfem.model_potential(b1.b, b2.b, CxxPtr(modelobj), false, false)
    collect(I)
end

end # module
