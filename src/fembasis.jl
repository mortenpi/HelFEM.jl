"""
    struct FEMBasis

Provides a Julia-based implementation of a finite element basis, but using HelFEM-provided
primitive polynomials. Unlike [`RadialBasis`](@ref), it does not scale the basis functions
with ``1/r``, and is therefore valid for any ``x`` interval on ``\\mathbb{R}``.

# Constructors

    FEMBasis(pb::PolynomialBasis, boundaries; [nquad])

Constructs a FEM basis with elements defined by the boundaries specified by `boundaries`
using the set of polynomials defined by `pb` as the primitive polynomials basis. Optionally,
`nquad` can be specified to change the number of quadrature points used in the Gauss-Chebyshev
quadrature when evaluating matrix elements.
"""
struct FEMBasis
    pb :: PolynomialBasis
    boundaries :: Vector{Float64}
    # Option whether to impose Dirichet 0 boundary conditions?
    bcdropleft :: Bool
    bcdropright :: Bool
    # Cache for numerical quadrature
    qxs :: Vector{Float64} # quadrature points on [-1, 1]
    qws :: Vector{Float64} # quadrature weights corresponding to qxs
    # Caches of basis polynomials and their derivatives evaluated at quadrature points qxs
    # Resulting in length(qxs) × length(pb) matrices
    pbf :: Matrix{Float64}
    pbdf :: Matrix{Float64}

    function FEMBasis(
            pb::PolynomialBasis, boundaries::AbstractVector;
            nquad=5*length(pb),
            # TODO: Implement actual support for not dropping the edge functions
            # bcdropleft = true, bcdropright = true,
        )
        @assert length(boundaries) >= 2
        # Check and sort the boundaries
        @assert all(isfinite.(boundaries))
        boundaries_sorted = collect(boundaries)
        sort!(boundaries_sorted)
        # Quadrature and primitive basis functions
        qxs, qws = chebyshev(nquad)
        pbf, pbdf = pb_eval(pb, qxs)
        # Construct the FEMBasis object
        b = new(pb, boundaries_sorted, true, true, #=TODO: bcdropleft, bcdropright,=# qxs, qws, pbf, pbdf)
        return b
    end
end

function Base.show(io::IO, b::FEMBasis)
    pbname = primbas_name(b.pb)
    pb_nnodes = nnodes(b.pb)
    nelem = nelements(b)
    x1, xn = first(b.boundaries), last(b.boundaries)
    nbf = length(b)
    # TODO: add introspection for BCs
    write(io, "FEMBasis($pbname($pb_nnodes nodes), $nelem elements, on [$x1, $xn]) with $nbf basis function$(nbf>1 ? "s" : "")")
end

function Base.length(b::FEMBasis)
    nelem = nelements(b)
    pb_length = length(b.pb)
    nbf = (pb_length - 1)*nelem + 1
    # Imposing a boundary condition on either left or right end of the interval will means
    # removing the left- or right-most function which is non-zero at the boundary.
    b.bcdropleft && (nbf -= 1)
    b.bcdropright && (nbf -= 1)
    return nbf
end

boundaries(b::FEMBasis) = ReadOnlyArray(b.boundaries)

nelements(b::FEMBasis) = length(b.boundaries) - 1

nquad(b::FEMBasis) = length(b.qxs)

function (b::FEMBasis)(qs; derivative=0)
    @assert derivative in [0, 1]
    @assert first(b.boundaries) <= minimum(qs)
    @assert last(b.boundaries) >= maximum(qs)
    bf = zeros(Float64, (length(qs), length(b)))
    pb_nbf = length(b.pb)
    # TODO: support _not_ dropping the edge functions
    for iel = 1:nelements(b)
        qmin, qmax = b.boundaries[iel], b.boundaries[iel+1]
        q0, qλ = (qmax + qmin) / 2, (qmax - qmin) / 2
        # We assign the r values on the boundaries to the element on the left. In other words,
        # we'll define the element to be r ∈ (rmin, rmax], except when iel == 1.
        idxs = (iel == 1) ? findall(q -> qmin <= q <= qmax, qs) : findall(q -> qmin < q <= qmax, qs)
        isempty(idxs) && continue
        # Scale the rs values in this element down to the [-1, 1] range
        xs = (qs[idxs] .- q0) ./ qλ
        # Evaluate basis functions (or their derivative, if dy is true) at xs
        ys = (derivative == 0) ? b.pb(xs) : pb_eval(b.pb, xs).dfs ./ qλ
        # Assign the calculated basis values to the correct place in the basis matrix
        bfrange_start = (iel == 1) ? 1 : (pb_nbf - 1) * (iel - 1)
        bfrange_end = (iel == nelements(b)) ? length(b) : (pb_nbf - 1) * iel
        ysrange_start = (iel == 1) ? 2 : 1
        ysrange_end = (iel == nelements(b)) ? (pb_nbf - 1) : pb_nbf
        # Assign them into the output matrix
        bf[idxs, bfrange_start:bfrange_end] .= ys[:, ysrange_start:ysrange_end]
    end
    return bf
end

function radial_integral(b::FEMBasis, f; lderivative=false, rderivative=false)
    Fij = zeros(Float64, (length(b), length(b)))
    pb_nbf = length(b.pb)
    # TODO: support _not_ dropping the edge functions
    for iel = 1:nelements(b)
        qmin, qmax = b.boundaries[iel], b.boundaries[iel+1]
        q0, qλ = (qmax + qmin) / 2, (qmax - qmin) / 2
        qs = b.qxs .* qλ .+ q0 # scale the quadrature points to q coordinate in element
        # Evaluate the function at quadrature points in this element
        fs = f.(qs)
        # Left and right function values
        Bi = lderivative ? b.pbdf : b.pbf
        Bj = rderivative ? b.pbdf : b.pbf
        # The integrals get a qλ^(1 - n - m) factor, where n and m are the orders of the
        # derivatives of the left and right basis functions. The reason is because we scale
        # the x axis in each element (by qλ), which means the derivative of the basis
        # function is 1/qλ times the derivative of the primitive polynomial. The additional
        # 1 comes from the integral itself (dq = qλ * dx).
        qλfact = qλ^(1 - (lderivative ? 1 : 0) - (rderivative ? 1 : 0))
        # Operator matrix in the element
        Fij_el = Bi' * (b.qws .* fs .* Bj) .* qλfact
        # Assign the calculated basis values to the correct place in the basis matrix
        bfrange_start = (iel == 1) ? 1 : (pb_nbf - 1) * (iel - 1)
        bfrange_end = (iel == nelements(b)) ? length(b) : (pb_nbf - 1) * iel
        ysrange_start = (iel == 1) ? 2 : 1
        ysrange_end = (iel == nelements(b)) ? (pb_nbf - 1) : pb_nbf
        # Assign them into the output matrix
        bfrange, ysrange = bfrange_start:bfrange_end, ysrange_start:ysrange_end
        Fij[bfrange, bfrange] .+= Fij_el[ysrange, ysrange]
    end
    return Fij
end

"""
    elementrange(b::RadialBasis, k::Integer) -> (r_k, r_{k+1})

Returns a tuple with the start and end boundary of the `k`-th element.
"""
function elementrange(b::FEMBasis, k::Integer)
    element_boundaries = boundaries(b)
    1 <= k < length(element_boundaries) || throw(DomainError(k, "Element index k out of range (1 <= k <= $(nelements(b)))"))
    element_boundaries[k], element_boundaries[k+1]
end

"""
    scale_to_element(b::RadialBasis, k::Integer, xs)

Scales the ``x`` values within the element `k` to the corresponding ``r`` coordinates.
"""
function scale_to_element(b::FEMBasis, k::Integer, xs)
    rmin, rmax = elementrange(b, k)
    r0, rλ = (rmax + rmin) / 2, (rmax - rmin) / 2
    xs .* rλ .+ r0
end

function controlpoints(b::FEMBasis)
    # We only know how to calculate control points for the LIP basis currently
    b.pb.primbas == 4 || throw(ArgumentError("Can only calculate control points for LIP basis"))
    # The LIPBasis uses Gauss-Lobatto nodes, so the following only applies to LIP
    pb_nnodes = nnodes(b.pb)
    xs, _ = lobatto(pb_nnodes) # we only need the nodes, can discard the weights
    rs = Vector{Float64}(undef, (pb_nnodes - 1)*nelements(b) + 1)
    for k in 1:nelements(b)
        rs[(pb_nnodes-1)*(k-1)+1:(pb_nnodes-1)*k+1] .= scale_to_element(b, k, xs)
    end
    return rs
end

function functionoverlap(b::FEMBasis, f; derivative=false)
    fi = zeros(Float64, length(b))
    pb_nbf = length(b.pb)
    # TODO: support _not_ dropping the edge functions
    for iel = 1:nelements(b)
        qmin, qmax = b.boundaries[iel], b.boundaries[iel+1]
        q0, qλ = (qmax + qmin) / 2, (qmax - qmin) / 2
        qs = b.qxs .* qλ .+ q0 # scale the quadrature points to q coordinate in element
        # Evaluate the function at quadrature points in this element
        fs = f.(qs)
        # Left and right function values
        Bi = derivative ? b.pbdf : b.pbf
        # The integrals get a qλ^(1 - n - m) factor, where n and m are the orders of the
        # derivatives of the left and right basis functions. The reason is because we scale
        # the x axis in each element (by qλ), which means the derivative of the basis
        # function is 1/qλ times the derivative of the primitive polynomial. The additional
        # 1 comes from the integral itself (dq = qλ * dx).
        qλfact = qλ^(1 - (derivative ? 1 : 0))
        # Operator matrix in the element
        fi_el = sum(b.qws .* fs .* Bi, dims=1) .* qλfact
        # Assign the calculated basis values to the correct place in the basis matrix
        bfrange_start = (iel == 1) ? 1 : (pb_nbf - 1) * (iel - 1)
        bfrange_end = (iel == nelements(b)) ? length(b) : (pb_nbf - 1) * iel
        ysrange_start = (iel == 1) ? 2 : 1
        ysrange_end = (iel == nelements(b)) ? (pb_nbf - 1) : pb_nbf
        # Assign them into the output matrix
        bfrange, ysrange = bfrange_start:bfrange_end, ysrange_start:ysrange_end
        fi[bfrange] .+= fi_el[ysrange]
    end
    return fi
end