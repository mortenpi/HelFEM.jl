# Polynomial bases

The [`HelFEM.PolynomialBasis`](@ref) type wraps the corresponding C++ class and defines a primitive polynomial basis --- a set of polynomials ``\{p_i(x)\}`` on the domain ``x \in [-1, 1]``.
In the finite element method, these are then repeated in each element.

Currently the library supports three types of polynomials:

* `:legendre`: Legendre polynomials
* `:hermite`: Hermite polynomials
* `:lip`: Lagrange interpolating polynomials (with Gauss-Lobatto nodes)

Each primitive basis is defined by the number of nodes, which then determines the order of the polynomials and how many polynomials are there in the set.

```@example
using HelFEM, Plots
xs = range(-1, 1, length=2001)
a = @animate for nnodes = 2:15
    b1 = PolynomialBasis(:legendre, nnodes)
    b2 = PolynomialBasis(:hermite, nnodes)
    b3 = PolynomialBasis(:lip, nnodes)
    plot(
        plot(xs, b1(xs), label=false, title="Legendre, nnodes=$nnodes"),
        plot(xs, b2(xs), label=false, title="Hermite, nnodes=$nnodes"),
        plot(xs, b3(xs), label=false, title="LIP, nnodes=$nnodes"),
        layout = (3, 1), size=(800, 1000)
    )
end
gif(a, fps=0.5)
```

Various methods are available to work with the polynomial basis and for introspection (see the reference).

## Reference

```@docs
PolynomialBasis
length(::PolynomialBasis)
HelFEM.nnodes(::PolynomialBasis)
```
