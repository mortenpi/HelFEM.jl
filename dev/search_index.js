var documenterSearchIndex = {"docs":
[{"location":"polynomial/#Polynomial-bases","page":"Polynomial bases","title":"Polynomial bases","text":"","category":"section"},{"location":"polynomial/","page":"Polynomial bases","title":"Polynomial bases","text":"using HelFEM, Plots\nlet xs = range(-1, 1, length=2001)\n    a = @animate for nnodes = 2:15\n        b = HelFEM.PolynomialBasis(:legendre, nnodes)\n        ys = b(xs)\n        plot(xs, ys, label=false, title=\"Legendre, nnodes=$nnodes\")\n    end\n    gif(a, fps=0.5)\nend","category":"page"},{"location":"polynomial/","page":"Polynomial bases","title":"Polynomial bases","text":"using HelFEM, Plots\nlet xs = range(-1, 1, length=2001)\n    a = @animate for nnodes = 2:15\n        b = HelFEM.PolynomialBasis(:hermite, nnodes)\n        ys = b(xs)\n        plot(xs, ys, label=false, title=\"Hermite, nnodes=$nnodes\")\n    end\n    gif(a, fps=0.5)\nend","category":"page"},{"location":"polynomial/","page":"Polynomial bases","title":"Polynomial bases","text":"using HelFEM, Plots\nlet xs = range(-1, 1, length=2001)\n    a = @animate for nnodes = 2:15\n        b = HelFEM.PolynomialBasis(:lip, nnodes)\n        ys = b(xs)\n        plot(xs, ys, label=false, title=\"LIP, nnodes=$nnodes\")\n    end\n    gif(a, fps=0.5)\nend","category":"page"},{"location":"polynomial/#Reference","page":"Polynomial bases","title":"Reference","text":"","category":"section"},{"location":"polynomial/","page":"Polynomial bases","title":"Polynomial bases","text":"PolynomialBasis","category":"page"},{"location":"polynomial/#HelFEM.PolynomialBasis","page":"Polynomial bases","title":"HelFEM.PolynomialBasis","text":"struct PolynomialBasis\n\nWrapper type for the helfem::atomic::polynomial_basis::PolynomialBasis class, representing a set of polynomials p_i(x) on a domain x in -1 1.\n\nConstructors\n\nPolynomialBasis(basis::Symbol, nnodes)\n\nConstructs a particular polynomial basis with a specific number of defining nodes. basis must be one of: :hermite, :legendre or :lip.\n\n\n\n\n\n","category":"type"},{"location":"radialbasis/#RadialBasis","page":"RadialBasis","title":"RadialBasis","text":"","category":"section"},{"location":"radialbasis/","page":"RadialBasis","title":"RadialBasis","text":"The helfem::atomic::basis::RadialBasis C++ class defines a set of basis functions b_n(r) on the domain r in 0 r_rmmax, where r_rmmax is a user-defined parameter. They are used to approximate functions on that domain via basis expansions","category":"page"},{"location":"radialbasis/","page":"RadialBasis","title":"RadialBasis","text":"f(r) = sum_n c_n b_n(r)","category":"page"},{"location":"radialbasis/","page":"RadialBasis","title":"RadialBasis","text":"The basis consists of a set of polynomial functions repeated on consecutive sub-elements of the domain. Specifically, the N elements are defined by a sorted list of N+1 element boundaries at r_k, with r_0 = 0 and r_N = r_rmmax.","category":"page"},{"location":"radialbasis/#Polynomials-and-elements","page":"RadialBasis","title":"Polynomials and elements","text":"","category":"section"},{"location":"radialbasis/","page":"RadialBasis","title":"RadialBasis","text":"The polynomials p_i(x) are represented by instances of subclasses of helfem::atomic::polynomial_basis::PolynomialBasis, wrapped in Julia with PolynomialBasis, and they are defined on the domain x in -1 1. For each element, the x-axis is scaled and shifted to cover the element. That is, by using the midpoint r^(k)_rmmid and half-width r^(k)_lambda of the element of the k-th element","category":"page"},{"location":"radialbasis/","page":"RadialBasis","title":"RadialBasis","text":"r^(k)_rmmid = fracr_k + r_k-12 quad\nr^(k)_lambda = fracr_k - r_k-12","category":"page"},{"location":"radialbasis/","page":"RadialBasis","title":"RadialBasis","text":"the x and r coordinates for each element can be transformed into each other with","category":"page"},{"location":"radialbasis/","page":"RadialBasis","title":"RadialBasis","text":"r = r^(k)_rmmid + x cdot r^(k)_lambda\nquadtextrmorquad\nx = left( r - r^(k)_rmmid right)  r^(k)_lambda","category":"page"},{"location":"radialbasis/","page":"RadialBasis","title":"RadialBasis","text":"Note that this is valid and meaningful only within an element. Or in other words, you must always have x in -1 1 and k picked appropriately such that r in r_k-1 r_k.","category":"page"},{"location":"radialbasis/","page":"RadialBasis","title":"RadialBasis","text":"We also assume that the first and the last polynomial is non-zero at the element boundary, whereas the other polynomials vanish at the element boundaries.","category":"page"},{"location":"radialbasis/#Basis-functions","page":"RadialBasis","title":"Basis functions","text":"","category":"section"},{"location":"radialbasis/","page":"RadialBasis","title":"RadialBasis","text":"The basis function b_(ki), corresponding to the i-th polynomials in the k-th element, is defined by","category":"page"},{"location":"radialbasis/","page":"RadialBasis","title":"RadialBasis","text":"b_(ki)(r) = p_i(r)  r\nquad\nforall r notin r_k-1 r_k  b_(ki)(r) = 0","category":"page"},{"location":"radialbasis/","page":"RadialBasis","title":"RadialBasis","text":"That is, they are the basis polynomials in the element, scaled by 1r, and zero outside of the element.","category":"page"},{"location":"radialbasis/","page":"RadialBasis","title":"RadialBasis","text":"That is, almost. There are two additional things to consider:","category":"page"},{"location":"radialbasis/","page":"RadialBasis","title":"RadialBasis","text":"Nodes at element boundaries. The polynomials are defined by nodes, and the edge nodes of each element are shared by two elements. For this reason, the basis functions that are non-zero at element boundaries r = r_k (corresponding to x = pm 1) are formally considered to be the same basis function, just spanning two elements. In other words, b_(k-1 M) and b_(k 1) (where M is the number of polynomials in each element) are glued together into a single function for k = 2 ldots N-1.\nBoundary conditions. In order to impose the boundary conditions r f(r) to 0 as r to 0 and f(r_rmmax) = 0, which come from the physical problem, the first function b_(11)(r) in the first element and the last function b_(NM)(r) in the last element are removed, since they would not obey the boundary conditions.","category":"page"},{"location":"radialbasis/","page":"RadialBasis","title":"RadialBasis","text":"Overall, this means that if you have N elements and M polynomials, you have N cdot M - (N-1) - 2 = N cdot (M - 1) - 1 basis functions.","category":"page"},{"location":"radialbasis/","page":"RadialBasis","title":"RadialBasis","text":"warning: TODO\nThis may not be entirely true – it looks like you can have more than one overlapping function between elements? C.f. Nbf method of the RadialBasis class:size_t RadialBasis::Nbf() const {\n  // Number of basis functions is Nprim*Nel - (Nel-1)*noverlap - 1 - noverlap\n  return Nel() * (bf.n_cols - poly->get_noverlap()) - 1;\n}What's poly->get_noverlap()?","category":"page"},{"location":"radialbasis/#Matrix-elements","page":"RadialBasis","title":"Matrix elements","text":"","category":"section"},{"location":"radialbasis/","page":"RadialBasis","title":"RadialBasis","text":"The radial basis set also provides methods to evaluate matrix elements of various operators.","category":"page"},{"location":"radialbasis/","page":"RadialBasis","title":"RadialBasis","text":"A_ij = int_0^r_rmmax b^*_i(r) hatA b_j(r)  r^2 dr","category":"page"},{"location":"radialbasis/","page":"RadialBasis","title":"RadialBasis","text":"The inner product, by definition, has a weight function of w(r) = r^2.","category":"page"},{"location":"radialbasis/","page":"RadialBasis","title":"RadialBasis","text":"The physical motivation for both the weight function and the 1r scaling of the basis polynomials comes from spherical symmetry, since we assume that we are solving spherically symmetric equations on mathbbR^3. The functions on mathbbR^3 are assumed to be represented using a radial function P(r) and spherical harmonics","category":"page"},{"location":"radialbasis/","page":"RadialBasis","title":"RadialBasis","text":"Psi(vecx) = P(r) Y_ell m(thetavarphi)","category":"page"},{"location":"radialbasis/","page":"RadialBasis","title":"RadialBasis","text":"and we will use the basis set to expand P(r). The integrals of spherically symmetric operators then get an r^2 weight from the angular integration when reduced down to a radial integral","category":"page"},{"location":"radialbasis/","page":"RadialBasis","title":"RadialBasis","text":"int_mathbbR^3 Psi^*_1(vecx) hatA Psi_2(vecx) dV\n= delta_ell_1ell_2 delta_m_1 m_2 int_0^infty P^*_1(r) hatA P_2(r)  r^2 dr","category":"page"},{"location":"radialbasis/","page":"RadialBasis","title":"RadialBasis","text":"By scaling the basis polynomials with 1r, we actually cancel out the weight function, and so the integrals for the matrix elements become just simple integrals over polynomials within an element","category":"page"},{"location":"radialbasis/","page":"RadialBasis","title":"RadialBasis","text":"A_(ki)(kj) = int_r_k-1^r_k p_i(x(r)) hatA p_j(x(r))  r^2 dr","category":"page"},{"location":"radialbasis/","page":"RadialBasis","title":"RadialBasis","text":"where x(r) takes care of the scaling between the x and r values for the element.","category":"page"},{"location":"radialbasis/","page":"RadialBasis","title":"RadialBasis","text":"When evaluating the matrix elements, HelFEM uses modified Gauss-Chebyshev quadrature. The number of quadrature points nquad needs to be defined by the user.","category":"page"},{"location":"radialbasis/#Examples","page":"RadialBasis","title":"Examples","text":"","category":"section"},{"location":"radialbasis/","page":"RadialBasis","title":"RadialBasis","text":"using HelFEM, Plots\npb = HelFEM.PolynomialBasis(:lip, 4)\nxgrid = [0, 1, 2, 3]\nb = HelFEM.RadialBasis(pb, xgrid, 100)\n# Calculate points for plotting\nrs = range(minimum(xgrid), maximum(xgrid), length=501)\nbs = b(rs)\na = @animate for i in 1:length(b)\n    plot(layout=(2,1), size=(900, 700), legend=false,\n        ylabel = [raw\"$b(r)$\" raw\"$r~b(r)$\"]\n    )\n    plot!(rs, bs, c=:lightgray, subplot=1)\n    plot!(rs, bs[:,i], c=:black, subplot=1)\n    plot!(rs, bs .* rs, c=:lightgray, subplot=2)\n    plot!(rs, bs[:,i] .* rs, c=:black, subplot=2)\n    vline!([xgrid xgrid], ls=:dash, c=1)\nend\ngif(a, fps=1)","category":"page"},{"location":"radialbasis/#Reference","page":"RadialBasis","title":"Reference","text":"","category":"section"},{"location":"radialbasis/","page":"RadialBasis","title":"RadialBasis","text":"RadialBasis","category":"page"},{"location":"radialbasis/#HelFEM.RadialBasis","page":"RadialBasis","title":"HelFEM.RadialBasis","text":"struct RadialBasis\n\nWrapper type for the helfem::atomic::basis::RadialBasis class, representing a finite element basis for radial functions on the domain r in 0 r_rmmax.\n\n\n\n\n\n","category":"type"},{"location":"#HelFEM.jl","page":"Overview","title":"HelFEM.jl","text":"","category":"section"},{"location":"","page":"Overview","title":"Overview","text":"HelFEM.jl is a Julia wrapper for the HelFEM finite element method library.","category":"page"}]
}
