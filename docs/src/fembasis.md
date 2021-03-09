# Finite element basis

The [`FEMBasis`](@ref) type provides an alternative implementation of the finite element
basis with slightly different conventions. Unlike [`RadialBasis`](@ref), it does not scale
the elements with ``1/r``, so it can also span into negative ``x`` values.

It is also implemented directly in Julia, rather than wrapping a C++ class, although it still
relies on the primitive polynomial bases from the HelFEM C++ library.)

### Elements, polynomials and basis functions

The basis is defined as a set of polynomials (``\{p_i(t)\}``; an instance of
[`PolynomialBasis`](@ref)) that is repeated in each element. The elements are simply defined
as a list of ``N+1`` real values ``x_1 < x_2 < \ldots < x_{N+1}``.

Each polynomial is defined on the domain ``t \in [-1, 1]`` and so for each element where
``x \in [x_i, x_{i+1}]``, the transformation between the two coordinates is given by

```math
x = x^{(k)}_{\rm{mid}} + x^{(k)}_{\rm{\lambda}} t
\iff
t = \frac{x - x^{(k)}_{\rm{mid}}}{x^{(k)}_{\rm{\lambda}}}
```

where ``x^{(k)}_{\rm{mid}}`` and ``x^{(k)}_{\rm{\lambda}}`` are the midpoint and width of
the ``k``-th element, respecively, defined by

```math
x^{(k)}_{\rm{mid}} = \frac{x_{i+1} + x_i}{2},
\quad
x^{(k)}_{\rm{\lambda}} = \frac{x_{i+1} - x_i}{2}
```

The basis functions ``b_i(x)`` are therefore just the original polynomials scaled to each
element using that transformation.
The only exception to that are the first and last polynomial in each element that are formally
joined together into a single basis function that spans two elements.
By definition, outside of the range ``[x_1, x_{N+1}]``, the basis functions are assumed to
be zero.

The basis also drops the first polynomial in the first elements and the last polynomial in
the last elements to impose the boundary conditions

```math
b_i(x_1) = 0, \quad b_i(x_N) = 0
```

```@example
using HelFEM, Plots
pb = PolynomialBasis(:lip, 4)
xgrid = [-2, -0.5, 1.5, 3]
b = FEMBasis(pb, xgrid)
# Calculate points for plotting
rs = range(minimum(xgrid), maximum(xgrid), length=501)
bs = b(rs)
a = @animate for i in 1:length(b)
    plot(size=(900, 350), legend=false, ylabel = [raw"$b(r)$" raw"$r~b(r)$"])
    plot!(rs, bs, c=:lightgray)
    plot!(rs, bs[:,i], c=:black)
    vline!([xgrid xgrid], ls=:dash, c=1)
end
gif(a, fps=1)
```

### Matrix elements

The [`radial_integral`](@ref radial_integral(::FEMBasis, ::Any)) method can be used to
evaluate matrix elements of a function ``f(x)`` between the basis functions (or its
derivatives)

```math
f^{(n,m)}_{ij} = \int_{x_1}^{x_{N+1}} b^{(n)*}_{i}(x) f(x) b^{(m)}_{j}(x) ~ dx
```

By default, the basis functions themselves are used (``n = m = 0``), but optionally it is
also possible to substitute them with derivatives (i.e. ``n = 1`` and/or ``m=1``).

## Reference

```@docs
FEMBasis
FEMBasis(::Any) # corresponds to (::FEMBasis)(::Any)
length(::FEMBasis)
boundaries(::FEMBasis)
nelements(::FEMBasis)
radial_integral(::FEMBasis, ::Any)
```

## Internal methods

These methods are not part of the public API, so they may change at any time.

```@docs
HelFEM.nquad(::FEMBasis)
HelFEM.elementrange(::FEMBasis, ::Integer)
HelFEM.scale_to_element(::FEMBasis, ::Integer, ::Any)
```
