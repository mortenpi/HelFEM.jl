# # 1D Particle in a Box
#
# This notebook demonstrates how to generically set up a simple eigenvalue
# calculation within the `ContinuumArrays` framework and then uses the HelFEM
# FEM basis to solve it.

using LinearAlgebra
using Plots
using DataFrames, DataFramesMeta, StatsPlots
using ContinuumArrays
using HelFEM
using HelFEM.CompactFEMBasis: HelFEMBasis
using QuasiArrays: domain, AbstractQuasiMatrix
## Set the default plot size to more or less fill the page width
default(size=(800, 400))

# The equation we use for this example is
#
# ```math
# \frac{d^2 f}{dx^2} = \lambda f(x)
# ```
#
# with ``x \in [a, b]``, i.e. under the boundary condition
#
# ```math
# f(a) = f(b) = 0
# ```
# The analytical eigenvalues and eigenfunctions of this problem are given by
#
# ```math
# \lambda_n = - \left(\frac{\pi n}{b - a}\right)^2,
# \quad\quad
# f_n(x) = \sqrt{\frac{2}{b - a}} \sin\left(
#   n \pi \frac{x - a}{b - a}
# \right)
# ```
#
# where ``n \in \mathbb{Z}^+``.
#
# We define the following Julia functions to access the analytical values. The `Basis`
# object is only necessary for specifying the domain (i.e. ``a`` and ``b``).

function λ(R::AbstractQuasiMatrix, n::Integer)
    @assert size(R, 1) == ℵ₁
    @assert n >= 1
    a, b = extrema(domain(axes(R, 1)))
    - ((π*n)/(b - a))^2
end

function f(R::AbstractQuasiMatrix, n::Integer, x)
    @assert size(R, 1) == ℵ₁
    @assert x ∈ axes(R, 1)
    @assert n >= 1
    a, b = extrema(domain(axes(R, 1)))
    sqrt(2/(b-a)) * sin((x-a)*n*π/(b - a))
end
nothing # hide

# The following function solves this numerically with a ContinuumArrays `Basis`.
# It returns the eigenvectors and eigenvalues of the operator within the ContinuumArrays
# framework, only requiring a `Basis` object to be passed.
# The boundaries (i.e. the domain) are again defined by the `Basis` object.

function solve_∇²(R::AbstractQuasiMatrix)
    ## We restrict R to AbstractQuasiMatrix, not Basis, because if you e.g. restrict a basis
    ## with R[:, 2:end-1] to impose boundary conditions, that will not return a Basis object,
    ## but it still a valid basis set.
    @assert size(R, 1) == ℵ₁
    ## The boundary box is determined from the domain of the basis.
    xmin, xmax = extrema(domain(axes(R, 1)))
    ## Construct the abstract derivative operators on [a, b].
    D = Derivative(axes(R, 1))
    ## Solve the generalized eigenvalue problem, since the basis, in general, will not be
    ## orthonormal. R'R yields the overlap matrix and -R'D'D*R materializes the matrix
    ## corresponding to the second derivative operator (note: D² = -D'D).
    L, S = collect(-(R'D'D*R)), collect(R'R)
    e = eigen(L, S)
    ## Just a sanity check to make sure that all the calculated eigenvalues are negative,
    ## which we assume based on the analytical solution.
    @assert all(<=(0), e.values)
    return (
        R = R,
        a = xmin, b = xmax,
        λ = reverse(e.values),
        states = reverse(e.vectors, dims=2),
    )
end
nothing # hide

# ### Helper functions
#
# Let's define a bunch of helper functions for plotting etc.

## Plot the numerical and corresponding analytical solution for a given n
function plot_cmp!(s, n; c=nothing, label=nothing, kwargs...)
    a, b = extrema(domain(axes(s.R, 1)))
    ## Set of point where the plot the functions
    xs = range(a, b, length=501)
    ## Evaluate the basis function at the x grid
    B = s.R[xs, :]
    ## Extract the n-th state. We normalize the sign of the function to start with a
    ## positive derivative on the left edge. We're assuming that the first basis function is
    ## the left-most one and that it is positive.
    v = s.states[:, n] * sign(s.states[1, n])
    ## We use the generalized eigenvalue solver, but the eigenvectors it produces are not
    ## normalized with respect to the inner product (that takes the overlap matrix into
    ## account). So we normalize the vector by hand.
    v ./= sqrt(dot(v, s.R's.R, v))
    ## Default color and label, if not explicitly passed
    isnothing(c) && (c = n)
    isnothing(label) && (label = "n = $n")
    ## Plot the numerical solution to the eigenfunction
    plot!(xs, B * v; c = c, ls=:solid, label=label, kwargs...)
    ## Evaluate and plot the corresponding analytic solution
    plot!(xs, (x -> f(s.R, n, x)).(xs); c = c, ls=:dash, label=nothing, kwargs...)
end

## Plot the numerical and analytical solutions for a whole set of n values on a single plot
function plot_cmp_ns(s, ns)
    plot(legend = :bottomleft)
    for n in ns
        plot_cmp!(s, n)
    end
    plot!([], ls=:solid, c = :black, label = "Numerical")
    plot!([], ls=:dash, c = :black, label = "Analytic")
end

## Turn a solution into a DataFrame
function eigenvalue_dataframe(s)
    ns = 1:length(s.λ)
    df = DataFrame(
        n = ns,
        λ = s.λ,
        ## Ref is necessary because R is not declared a scalar in terms of broadcasting
        λ_exact = λ.(Ref(s.R), ns),
    )
    return @transform(df, δ = :λ .- :λ_exact)
end
nothing # hide

# ## Example solution
#
# Let's construct a very small basis (that's not going to be very good) and solve the
# equation in that basis.

R = let a = -3, b = 5
    pb = PolynomialBasis(:lip, 4)
    b = FEMBasis(pb, [a, a + (b-a)*0.33, b])
    HelFEMBasis(b)
end
s = solve_∇²(R)
nothing # hide

# The solutions for the three lowest eigenvalues look like this.

plot_cmp_ns(s, 1:3)

# For the higher ``n`` ones we see singular points at the element boundary.

plot_cmp_ns(s, 4:5)
vline!(boundaries(R.b), c = :gray, ls=:dash, lw=2, label="FEM boundaries")

# The eigenvalues and their differences from the analytic values are

Λ = eigenvalue_dataframe(s)

# And we can visualize the error, both on a linear and semi-log scale

plot(
    @df(Λ, scatter(abs.(:δ), ylabel = "Diff. from analytic eigenvalue")),
    @df(Λ, scatter(abs.(:δ), yaxis=:log10)),
    legend = false, xlabel = "Eigenvalue index n",
)

# ### Finer basis
#
# Let's now do the same exercise, but for a finer basis set.

R = let a = -3, b = 5
    pb = PolynomialBasis(:lip, 4)
    b = FEMBasis(pb, range(a, b, length=31))
    HelFEMBasis(b)
end
s = solve_∇²(R)
nothing # hide

#-

Λ = eigenvalue_dataframe(s)
Λ[1:5, :]

#-

plot(
    @df(Λ, scatter(abs.(:δ), ylabel = "Diff. from analytic eigenvalue")),
    @df(Λ, scatter(abs.(:δ), yaxis=:log10)),
    legend = false, xlabel = "Eigenvalue index n",
)

# The orbitals match the analytical solutions almost perfectly.

plot_cmp_ns(s, 1:3)

# Even at higher ``n`` values it is still quite fine.

plot_cmp_ns(s, 30)

# For fun, let's look at the last 12 solution we get numerically and compare them to what
# we would expect to get analytically. As one would expect, the functions do not really
# match the analytical solutions.

ps = [
    plot_cmp_ns(s, lastindex(s.λ) - i)
    for i = 0:(3*4) - 1
]
plot(ps..., layout=(4, 3), size = (800, 750))

# ## Convergence of eigenvalues
#
# Here we calculate the eigenvalues for a set of different basis sets, increasing it in a
# systematic way (in this case, diving the interval into more and more elements).

Λs = map(2 .^ (0:9)) do N
    R = let a = -3, b = 5
        pb = PolynomialBasis(:lip, 4)
        b = FEMBasis(pb, range(a, b, length = round(Int, N) + 1))
        HelFEMBasis(b)
    end
    s = solve_∇²(R)
    df = eigenvalue_dataframe(s)
    df[!, :N] .= N
    df[!, :nb] .= size(R, 2)
    return df
end |> dfs -> vcat(dfs...)
describe(Λs)

# The errors of the eigenvalues converge systematically to the true value (or at least until
# they reach floating point roundoff errors).

@df @where(Λs, 1 .<= :n .<= 5) plot(
    :N, abs.(:δ), group=:n,
    yaxis=:log10, xaxis=:log10, m=:dot
)
xticks!(2 .^ (0:9), string.(2 .^ (0:8)))
xlabel!("Number of FEM elements")
ylabel!("Error of the n-th eigenvalue")

# ## Linear spline basis
#
# The ContinuumArrays package defines a simple basis of linear splines. So, to demonstrate
# that the `solve_∇²` function is independent of the basis set, let's use the linear splines
# to also solve the equation.
#
# The basis looks like a bunch of overlapping triangles.

B = let
    B = Spline{1}(range(-3, 5, length=11))
    ## Impose Dirichlet-0 boundary conditions, i.e. f(a) = f(b) = 0
    B[:, 2:end-1]
end
## Let's plot the basis functions
let
    a, b = extrema(domain(axes(B, 1)))
    xs = range(a, b, length=500)
    b = B[xs, :]
    plot(xs, b, legend=false)
end

# We can pass the basis into the `solve_∇²` function and it will yield a reasonable, even if
# a little jagged, solution for the lowest eigenfunctions

s = solve_∇²(B)
plot_cmp_ns(s, 1:3)

# Let's also see what the convergence looks like with linear splines:

Λs = map(2 .^ (1:12)) do N
    R = let a = -3, b = 5
        B = Spline{1}(range(a, b, length=N))
        ## Impose Dirichlet-0 boundary conditions, i.e. f(a) = f(b) = 0
        B[:, 2:end-1]
    end
    s = solve_∇²(R)
    df = eigenvalue_dataframe(s)
    df[!, :N] .= N
    df[!, :nb] .= size(R, 2)
    return df
end |> dfs -> vcat(dfs...)
describe(Λs)

#-

@df @where(Λs, 1 .<= :n .<= 5) plot(
    :N, abs.(:δ), group=:n,
    yaxis=:log10, xaxis=:log10, m=:dot
)
xticks!(2 .^ (2:12), string.(2 .^ (2:12)))
xlabel!("Number linear splines")
ylabel!("Error of the n-th eigenvalue")
