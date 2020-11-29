module HelFEM
using CxxWrap: CxxPtr

export PolynomialBasis, RadialBasis

module helfem
    using CxxWrap, HelFEM_jll
    @wrapmodule(libhelfem)
    function __init__()
        @initcxx
        verbose(false)
    end

    function ArmaVector(v::AbstractVector)
        av = ArmaVector(length(v))
        for (i, x) in enumerate(v)
            at!(av, i - 1, convert(Float64, x))
        end
        return av
    end
    Base.size(v::ArmaVector) = (Int(nrows(v)),)
    function Base.size(v::ArmaVector, dim::Integer)
        dim <= 0 && throw(ArgumentError("dimension $dim out of range"))
        dim == 1 ? Int(nrows(v)) : 1
    end
    Base.collect(v::ArmaVector) = [at(v, i - 1) for i = 1:size(v, 1)]

    function ArmaMatrix(M::AbstractMatrix)
        aM = ArmaMatrix(size(M, 1), size(M, 2))
        idxs = CartesianIndices(M)
        for (k, x) in enumerate(M)
            i, j = Tuple(idxs[k])
            at!(aM, i - 1, j - 1, convert(Float64, x))
        end
        return aM
    end
    Base.size(m::ArmaMatrix) = (Int(nrows(m)), Int(ncols(m)))
    function Base.size(m::ArmaMatrix, dim::Integer)
        dim <= 0 && throw(ArgumentError("dimension $dim out of range"))
        dim == 1 ? Int(nrows(m)) : dim == 2 ? Int(ncols(m)) : 1
    end
    Base.collect(m::ArmaMatrix) = [at(m, i - 1, j - 1) for i = 1:size(m, 1), j = 1:size(m, 2)]
end

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

## Primitive polynomials bases (PolynomialBasis)

@doc raw"""
    struct PolynomialBasis

Wrapper type for the `helfem::atomic::polynomial_basis::PolynomialBasis` class, representing
a set of polynomials ``\{p_i(x)\}`` on a domain ``x \in [-1, 1]``.
"""
struct PolynomialBasis
    pb :: helfem.PolynomialBasis
    primbas :: Int

    function PolynomialBasis(basis::Symbol, nnodes::Integer)
        primbas = (basis == :hermite) ? 2 :
                  (basis == :legendre) ? 3 :
                  (basis == :lip) ? 4 : error("Invalid primitive basis name $basis")
        new(helfem.polynomial_basis(primbas, nnodes), primbas)
    end
end

primbas_name(primbas::Int) =
    (primbas == 2) ? "HermiteBasis" :
    (primbas == 3) ? "LegendreBasis" :
    (primbas == 4) ? "LIPBasis" : error("Invalid primbas value $primbas")

function Base.show(io::IO, pb::PolynomialBasis)
    classname = primbas_name(pb.primbas)
    order = helfem.pb_order(pb.pb)
    nbf = length(pb)
    write(io, "PolynomialBasis($(primbas_name(pb.primbas)), order=$order) with $(nbf) basis functions")
end

Base.length(pb::PolynomialBasis) = helfem.get_nbf(pb.pb)

function (pb::PolynomialBasis)(xs)
    arma_xs = helfem.ArmaVector(collect(xs))
    return collect(helfem.pb_eval(pb.pb, arma_xs))
end

@doc raw"""
    struct RadialBasis

Wrapper type for the `helfem::atomic::basis::RadialBasis` class, representing a finite
element basis for radial functions on the domain ``r \in [0, r_{\rm{max}}]``.
"""
struct RadialBasis
    b :: helfem.RadialBasis
    nnodes :: Int
    nelem :: Int
    primbas :: Int
    rmax :: Float64
    igrid :: Int
    zexp :: Float64
    nquad :: Int

    function RadialBasis(nnodes, nelem; primbas = 4, rmax = 40.0, igrid = 4, zexp = 2.0, nquad = nothing)
        nquad = isnothing(nquad) ? 0 : nquad
        new(
            helfem.basis(nnodes, nelem, primbas, rmax, igrid, zexp, nquad),
            nnodes, nelem, primbas, rmax, igrid, zexp, nquad,
        )
    end

    function RadialBasis(pb::PolynomialBasis, grid::AbstractVector, nquad::Integer)
        b = helfem.RadialBasis(CxxPtr(pb.pb), nquad, helfem.ArmaVector(grid))
        nnodes = helfem.pb_order(pb.pb)
        nelem = length(grid) - 1
        primbas = pb.primbas
        rmax = last(grid)
        igrid = 0 # let's declare the igrid == 0 means a custom grid
        zexp = NaN
        new(b, nnodes, nelem, primbas, rmax, igrid, zexp, nquad)
    end
end

# Special RadialBasis constructor that allows you to construct a new RadialBasis based on an
# existing one, where you have added (or, also, removed) elements without changing the
# values of the first elements.
#
# TODO: add tests for this
function HelFEM.RadialBasis(b::RadialBasis; nelem)
    b_new = HelFEM.RadialBasis(b.nnodes, nelem;
        primbas = b.primbas, igrid = b.igrid, zexp = b.zexp, nquad = b.nquad,
        # calculated from the new nelem value:
        rmax = ^(1 + b.rmax, (nelem/b.nelem)^b.zexp) - 1,
    )
    imax = min(nelem, b.nelem) + 1
    @assert HelFEM.boundaries(b)[1:imax] ≈ HelFEM.boundaries(b_new)[1:imax]
    return b_new
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
        collect(S), collect(HelFEM.invh(S))
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

"""
Returns a vector of grid boundaries.
"""
function radialgrid(gridtype, nelem, rmax; zexp=nothing)
    igrid = (gridtype == :linear) ? 1 :
        (gridtype == :quadratic) ? 2 :
        (gridtype == :polynomial) ? 3 :
        (gridtype == :exponential) ? 4 :
        throw(ArgumentError("Invalid gridtype $gridtype"))
    if (gridtype == :linear || gridtype == :quadratic) && !isnothing(zexp)
        throw(ArgumentError("zexp has no effect on linear or quadratic grids"))
    end
    if (gridtype == :polynomial || gridtype == :exponential) && isnothing(zexp)
        throw(ArgumentError("Must set zexp for polynomial and exponential grids"))
    end
    collect(helfem.get_grid(rmax, nelem, igrid, isnothing(zexp) ? 0.0 : zexp))
end

function (b::RadialBasis)(rs)
    element_boundaries = boundaries(b)
    @assert minimum(element_boundaries) == 0
    @assert minimum(element_boundaries) <= minimum(rs)
    @assert maximum(element_boundaries) >= maximum(rs)
    # Group the rs values by by element. Note that we'll use the C++ indexing convention,
    # i.e. first element is labelled with 0.
    bf = zeros(Float64, (length(rs), length(b)))
    pb_ptr = helfem.get_poly(b.b)
    poly_nbf = helfem.get_nbf(pb_ptr)
    for iel = 0:(b.nelem - 1)
        rmin, rmax = element_boundaries[iel+1], element_boundaries[iel+2]
        r0, rλ = (rmax + rmin) / 2, (rmax - rmin) / 2
        # We assign the r values on the boundaries to the element on the left. In other words,
        # we'll define the element to be r ∈ (rmin, rmax]. This excludes r=0, but that is
        # intentional --  we need to special case that because of the 1/r scaling.
        idxs = findall(r -> rmin < r <= rmax, rs)
        isempty(idxs) && continue
        # Scale the rs values in this element down to the [-1, 1] range
        xs = (rs[idxs] .- r0) ./ rλ
        # Calculate the basis values in the element
        ys = collect(helfem.pb_eval(pb_ptr, helfem.ArmaVector(xs)))
        # Assign the calculated basis values to the correct place in the basis matrix
        bfrange_start = (iel == 0) ? 1 : (poly_nbf - 1) * iel
        bfrange_end = (iel == b.nelem - 1) ? length(b) : (poly_nbf - 1) * (iel + 1)
        ysrange_start = (iel == 0) ? 2 : 1
        ysrange_end = (iel == b.nelem - 1) ? (poly_nbf - 1) : poly_nbf
        # We also divide all the basis values by r, because that is how RadialBasis is defined
        # in HelFEM.
        bf[idxs, bfrange_start:bfrange_end] .= ys[:, ysrange_start:ysrange_end] ./ rs[idxs]
    end
    # Handle the r = 0 case, for which we need to look at the derivative of the polynomial
    # on the edge. That is, if f(0) = 0, then f(x)/x|_{x=0} = f'(0).
    idxs = findall(isequal(0), rs)
    if length(idxs) > 0
        zero_r = helfem.ArmaVector([-1]) # r = 0, therefore x = -1 in the first element
        ys, dys = helfem.ArmaMatrix(1, poly_nbf), helfem.ArmaMatrix(1, poly_nbf)
        helfem.pb_eval(pb_ptr, zero_r, ys, dys)
        dys = collect(dys)
        # If any of the r values are zero, we know that they will belong to the first
        # element, so we need to remove the leftmost basis function.
        for i in idxs
            # The 2/element_boundaries[2] factor comes from the scaling of the derivative
            # due to the scaling of the x-axis when going from [-1, 1] to [0, r_1].
            bf[i, 1:(poly_nbf - 1)] .= dys[1, 2:end] .* (2/element_boundaries[2])
        end
    end
    return bf
end

"""
    elementrange(b::RadialBasis, k::Integer) -> (r_k, r_{k+1})

Returns a tuple with the start and end boundary of the `k`-th element.
"""
function elementrange(b::RadialBasis, k::Integer)
    element_boundaries = boundaries(b)
    @assert 1 <= k < length(element_boundaries) # TODO: throw proper error
    element_boundaries[k], element_boundaries[k+1]
end

"""
    scale_to_element(b::RadialBasis, k::Integer, xs)

Scales the ``x`` values within the element `k` to the corresponding ``r`` coordinates.
"""
function scale_to_element(b::RadialBasis, k::Integer, xs)
    rmin, rmax = elementrange(b, k)
    r0, rλ = (rmax + rmin) / 2, (rmax - rmin) / 2
    xs .* rλ .+ r0
end

end # module
