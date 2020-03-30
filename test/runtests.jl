using HelFEM
using Test
using LinearAlgebra: I

@testset "HelFEM.jl" begin
    @testset "helfem" begin
        using HelFEM: helfem
        b = helfem.basis(10, 20, 4, 40, 2, 2.0, 0)
        m = helfem.radial_integral(b, b, 0, false, false)
        @test size(m, 1) === 179
        @test size(m, 2) === 179
        @test size(m, 3) === 1
        @test size(m) == (179, 179)
        @test_throws ArgumentError size(m, 0)
        @test_throws ArgumentError size(m, -100)
    end

    b1 = HelFEM.RadialBasis(16, 10)
    b2 = HelFEM.RadialBasis(6, 170)

    # Each element has the same number of basis functions as the order of the polynomials
    # (nnodes), but then we merge the ones on the element boundaries together and also
    # remove one function from each edge to satisfy boundary conditions
    nbf(nnodes, nelems) = nnodes * nelems - (nelems - 1) - 2
    @test length(b1) == nbf(16, 10)
    @test length(b2) == nbf(6, 170)

    S1 = HelFEM.overlap(b1)
    @test size(S1) == (length(b1), length(b1))
    S2, S2invh = HelFEM.overlap(b2, invh=true)
    @test size(S2) == (length(b2), length(b2))
    @test size(S2invh) == (length(b2), length(b2))

    @test S2invh' * S2 * S2invh ≈ I
    let S2invh_julia = sqrt(inv(S2))
        @test isreal(S2invh_julia)
        S2invh_julia = real(S2invh_julia)
        @test S2invh_julia' * S2 * S2invh_julia ≈ I
    end

    S1_rint = HelFEM.radial_integral(b1, 0)
    @test S1_rint ≈ S1
    S1_rint = HelFEM.radial_integral(b1, 0, b1)
    @test S1_rint ≈ S1

    S12_rint = HelFEM.radial_integral(b1, 0, b2)
    @test size(S12_rint) == (length(b1), length(b2))
end
