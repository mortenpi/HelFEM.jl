## Primitive polynomials bases (PolynomialBasis)

@doc raw"""
    struct PolynomialBasis

Wrapper type for the `helfem::atomic::polynomial_basis::PolynomialBasis` class, representing
a set of polynomials ``\{p_i(x)\}`` on a domain ``x \in [-1, 1]``.

# Constructors

    PolynomialBasis(basis::Symbol, nnodes)

Constructs a particular polynomial basis with a specific number of defining nodes.
`basis` must be one of: `:hermite`, `:legendre` or `:lip`.
"""
struct PolynomialBasis
    pb :: helfem.PolynomialBasis
    primbas :: Int
    nnodes :: Int

    function PolynomialBasis(basis::Symbol, nnodes::Integer)
        primbas = (basis == :hermite) ? 2 :
                  (basis == :legendre) ? 3 :
                  (basis == :lip) ? 4 : error("Invalid primitive basis name $basis")
        new(helfem.polynomial_basis(primbas, nnodes), primbas, nnodes)
    end
end

primbas_name(primbas::Int) =
    (primbas == 2) ? "HermiteBasis" :
    (primbas == 3) ? "LegendreBasis" :
    (primbas == 4) ? "LIPBasis" : error("Invalid primbas value $primbas")

primbas_name(pb::PolynomialBasis) = primbas_name(pb.primbas)

function Base.show(io::IO, pb::PolynomialBasis)
    classname = primbas_name(pb.primbas)
    order = helfem.pb_order(pb.pb)
    nbf = length(pb)
    write(io, "PolynomialBasis($(primbas_name(pb.primbas)), order=$order) with $(nbf) basis functions")
end

"""
    length(pb::PolynomialBasis) -> Integer

Returns the number of basis functions (polynomials) represented by the given polynomial basis.
"""
Base.length(pb::PolynomialBasis) = helfem.get_nbf(pb.pb)


"""
    HelFEM.nnodes(pb::PolynomialBasis) -> Integer

Returns the number of control nodes used to define the polynomial basis in the constructor.
"""
nnodes(pb::PolynomialBasis) = pb.nnodes

function (pb::PolynomialBasis)(xs)
    arma_xs = helfem.ArmaVector(collect(xs))
    return collect(helfem.pb_eval(pb.pb, arma_xs))
end

function pb_eval(pb::PolynomialBasis, xs::AbstractVector)
    npts, nbf = length(xs), length(pb)
    fs, dfs = helfem.ArmaMatrix(npts, nbf), helfem.ArmaMatrix(npts, nbf)
    helfem.pb_eval(pb.pb, helfem.ArmaVector(xs), fs, dfs)
    (fs = collect(fs), dfs = collect(dfs))
end

function controlpoints(pb::PolynomialBasis)
    # We only know how to calculate control points for the LIP basis currently
    pb.primbas == 4 || throw(ArgumentError("Can only calculate control points for LIP basis"))
    # The LIPBasis uses Gauss-Lobatto nodes, so the following only applies to LIP
    xs, _ = lobatto(pb.nnodes) # we only need the nodes, can discard the weights
    return xs
end
