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

    function RadialBasis(nnodes, nelem; primbas = 4, rmax = 40.0, igrid = 4, zexp = 2.0, nquad = nothing)
        new(helfem.basis(nnodes, nelem, primbas, rmax, igrid, zexp, isnothing(nquad) ? 0 : nquad))
    end
end

Base.length(b::RadialBasis) = Int(helfem.nbf(b.b))

function Base.show(io::IO, b::RadialBasis)
    nbf = helfem.nbf(b.b)
    nel = helfem.nel(b.b)
    write(io, "RadialBasis: $nbf basis functions ($nel elements)")
end

nquad(b::RadialBasis) = helfem.get_nquad(b.b)

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

boundaries(b::RadialBasis) = collect(helfem.get_bval(b.b))
add_boundary!(b::RadialBasis, r::Real) = helfem.add_boundary(b.b, r)

function quadraturepoints(b::RadialBasis)
    nq, nel = nquad(b), Int(helfem.nel(b.b))
    rs = Vector{Float64}(undef, nel*nq)
    for i = 0:nel-1
        el_rs = collect(helfem.get_r(b.b, i))
        @assert length(el_rs) == nq
        rs[nq*i+1:nq*(i+1)] .= el_rs
    end
    return rs
end

function quadratureweights(b::RadialBasis)
    nq, nel = nquad(b), Int(helfem.nel(b.b))
    rs = Vector{Float64}(undef, nel*nq)
    for i = 0:nel-1
        el_rs = collect(helfem.get_wrad(b.b, i))
        @assert length(el_rs) == nq
        rs[nq*i+1:nq*(i+1)] .= el_rs
    end
    return rs
end

function basisvalues(b::RadialBasis)
    nq, nel, nbf = nquad(b), Int(helfem.nel(b.b)), length(b)
    ys = zeros(nel*nq, nbf)
    nbf_count = 0
    for i = 0:nel-1
        el_ys = collect(helfem.get_bf(b.b, i))
        @assert size(el_ys, 1) == nq
        #@info "el=$i" size(el_ys)
        nbf_in_element = size(el_ys, 2)
        rs_range = nq*i+1 : nq*(i+1)
        bf_range_start = nbf_count == 0 ? 1 : nbf_count
        bf_range = bf_range_start:(bf_range_start+nbf_in_element-1)
        #@show bf_range
        ys[rs_range, bf_range] .= el_ys
        nbf_count = last(bf_range)
    end
    @assert nbf_count == nbf
    return ys
end

end # module
