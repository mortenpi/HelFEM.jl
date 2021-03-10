# RadialBasis

The `helfem::atomic::basis::RadialBasis` C++ class defines a set of basis functions ``b_n(r)`` on the domain ``r \in [0, r_{\rm{max}}]``, where ``r_{\rm{max}}`` is a user-defined parameter.
They are used to approximate functions on that domain via basis expansions

```math
f(r) = \sum_n c_n b_n(r)
```

The basis consists of a set of polynomial functions repeated on consecutive sub-elements of the domain.
Specifically, the ``N`` elements are defined by a sorted list of ``N+1`` _element boundaries_ at ``r_k``, with ``r_0 = 0`` and ``r_N = r_{\rm{max}}``.

### Polynomials and elements
The polynomials ``\{p_i(x)\}`` are represented by instances of subclasses of `helfem::atomic::polynomial_basis::PolynomialBasis`, wrapped in Julia with [`PolynomialBasis`](@ref), and they are defined on the domain ``x \in [-1, 1]``.
For each element, the ``x``-axis is scaled and shifted to cover the element.
That is, by using the midpoint ``r^{(k)}_{\rm{mid}}`` and half-width ``r^{(k)}_{\lambda}`` of the element of the ``k``-th element

```math
r^{(k)}_{\rm{mid}} = \frac{r_k + r_{k-1}}{2}, \quad
r^{(k)}_{\lambda} = \frac{r_k - r_{k-1}}{2}
```

the ``x`` and ``r`` coordinates for each element can be transformed into each other with

```math
r = r^{(k)}_{\rm{mid}} + x \cdot r^{(k)}_{\lambda}
\quad\textrm{or}\quad
x = \left( r - r^{(k)}_{\rm{mid}} \right) / r^{(k)}_{\lambda}
```

Note that this is valid and meaningful only within an element.
Or in other words, you must always have ``x \in [-1, 1]`` and ``k`` picked appropriately such that ``r \in [r_{k-1}, r_k]``.

We also assume that the first and the last polynomial is non-zero at the element boundary, whereas the other polynomials vanish at the element boundaries.

### Basis functions
The basis function ``b_{(k,i)}``, corresponding to the ``i``-th polynomials in the ``k``-th element, is defined by

```math
b_{(k,i)}(r) = p_{i}(r) / r,
\quad
\forall r \notin [r_{k-1}, r_k]: ~ b_{(k,i)}(r) = 0
```

That is, they are the basis polynomials in the element, scaled by ``1/r``, and zero outside of the element.

That is, almost. There are two additional things to consider:

1. **Nodes at element boundaries.**
   The polynomials are defined by nodes, and the edge nodes of each element are shared by two elements.
   For this reason, the basis functions that are non-zero at element boundaries ``r = r_k`` (corresponding to ``x = \pm 1``) are formally considered to be the same basis function, just spanning two elements.
   In other words, ``b_{(k-1, M)}`` and ``b_{(k, 1)}`` (where ``M`` is the number of polynomials in each element) are glued together into a single function for ``k = 2, \ldots, N-1``.

2. **Boundary conditions.**
   In order to impose the boundary conditions ``r f(r) \to 0`` as ``r \to 0`` and ``f(r_{\rm{max}}) = 0``, which come from the physical problem, the first function ``b_{(1,1)}(r)`` in the first element and the last function ``b_{(N,M)}(r)`` in the last element are removed, since they would not obey the boundary conditions.

Overall, this means that if you have ``N`` elements and ``M`` polynomials, you have ``N \cdot M - (N-1) - 2 = N \cdot (M - 1) - 1`` basis functions.

!!! warning "TODO"

    This may not be entirely true -- it looks like you can have more than one overlapping function between elements? C.f. `Nbf` method of the `RadialBasis` class:

    ```cpp
    size_t RadialBasis::Nbf() const {
      // Number of basis functions is Nprim*Nel - (Nel-1)*noverlap - 1 - noverlap
      return Nel() * (bf.n_cols - poly->get_noverlap()) - 1;
    }
    ```

    What's `poly->get_noverlap()`?

### Matrix elements

The radial basis set also provides methods to evaluate matrix elements of various operators.

```math
A_{ij} = \int_0^{r_{\rm{max}}} b^*_i(r) \hat{A} b_j(r) ~ r^2 dr
```

The inner product, by definition, has a weight function of ``w(r) = r^2``.

The physical motivation for both the weight function and the ``1/r`` scaling of the basis polynomials comes from spherical symmetry, since we assume that we are solving spherically symmetric equations on ``\mathbb{R}^3``.
The functions on ``\mathbb{R}^3`` are assumed to be represented using a radial function ``P(r)`` and spherical harmonics

```math
\Psi(\vec{x}) = P(r) Y_{\ell m}(\theta,\varphi)
```

and we will use the basis set to expand ``P(r)``. The integrals of spherically symmetric operators then get an ``r^2`` weight from the angular integration when reduced down to a radial integral

```math
\int_{\mathbb{R}^3} \Psi^*_1(\vec{x}) \hat{A} \Psi_2(\vec{x}) dV
= \delta_{\ell_1\ell_2} \delta_{m_1 m_2} \int_0^{\infty} P^*_1(r) \hat{A} P_2(r) ~ r^2 dr
```

By scaling the basis polynomials with ``1/r``, we actually cancel out the weight function, and so the integrals for the matrix elements become just simple integrals over polynomials within an element

```math
A_{(k,i),(k,j)} = \int_{r_{k-1}}^{r_k} p_i(x(r)) \hat{A} p_j(x(r)) ~ r^2 dr
```
where ``x(r)`` takes care of the scaling between the ``x`` and ``r`` values for the element.

When evaluating the matrix elements, HelFEM uses modified Gauss-Chebyshev quadrature.
The number of quadrature points `nquad` needs to be defined by the user.

## Examples

```@example
using HelFEM, Plots
pb = HelFEM.PolynomialBasis(:lip, 4)
xgrid = [0, 1, 2, 3]
b = HelFEM.RadialBasis(pb, xgrid, 100)
# Calculate points for plotting
rs = range(minimum(xgrid), maximum(xgrid), length=501)
bs = b(rs)
a = @animate for i in 1:length(b)
    plot(layout=(2,1), size=(900, 700), legend=false,
        ylabel = [raw"$b(r)$" raw"$r~b(r)$"]
    )
    plot!(rs, bs, c=:lightgray, subplot=1)
    plot!(rs, bs[:,i], c=:black, subplot=1)
    plot!(rs, bs .* rs, c=:lightgray, subplot=2)
    plot!(rs, bs[:,i] .* rs, c=:black, subplot=2)
    vline!([xgrid xgrid], ls=:dash, c=1)
end
gif(a, fps=1)
```

## Reference

```@docs
RadialBasis
```
