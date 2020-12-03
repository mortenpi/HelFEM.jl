# Particle in a box
using Plots
using LinearAlgebra: eigen
using HelFEM: HelFEM

a, b = -33.3, 7.1
pb = HelFEM.PolynomialBasis(:lip, 10)
B = HelFEM.FEMBasis(pb, range(a, b, length=21))
DD = HelFEM.radial_integral(B, q->1.0, lderivative=true, rderivative=true)
S = HelFEM.radial_integral(B, q->1.0)
e = eigen(DD, S)
# Compare energies with analytical energies (En = (πn/L)^2)
N=length(B)
es = (1:N) .^ 2 .* (π / (b - a))^2
e.values .- es
# Plot the wavefunctions -- they should be sinewaves:
xs = range(a, b, length=1001)
Bri = B(xs)
ns = 1:10
plot(xs, Bri*e.vectors[:,ns], alpha=(ns .^ -0.6)', legend=false)
