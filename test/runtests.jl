using HelFEM
using Test
using LinearAlgebra: I
using SparseArrays: SparseVector, SparseMatrixCSC

@testset "HelFEM.jl" begin
    @testset "helfem" begin
        using HelFEM: helfem
        b = helfem.basis(10, 20, 4, 40, 2, 2.0, 0)
        @test helfem.get_nquad(b) == 50
        @test helfem.get_poly_id(b) == 4
        @test helfem.get_poly_order(b) == 10
        m = helfem.radial_integral(b, b, 0, false, false)
        @test size(m, 1) === 179
        @test size(m, 2) === 179
        @test size(m, 3) === 1
        @test size(m) == (179, 179)
        @test_throws ArgumentError size(m, 0)
        @test_throws ArgumentError size(m, -100)

        # Wrapping of arma::vec and arma::mat
        @test collect(helfem.ArmaVector([1,2,3])) == [1,2,3]
        @test collect(helfem.ArmaMatrix([1 2 3; 4 5 6])) == [1 2 3; 4 5 6]
        # Make sure we accept general vectors and matrices
        @test collect(helfem.ArmaVector(SparseVector([1,0,0,2]))) == [1,0,0,2]
        @test collect(helfem.ArmaMatrix(SparseMatrixCSC([1 0 0; 0 5 0]))) == [1 0 0; 0 5 0]
    end

    b1 = HelFEM.RadialBasis(16, 10)
    b2 = HelFEM.RadialBasis(6, 170; rmax = 50.0)

    # Make sure that our basis function evaluation functions give the same results.
    b1(HelFEM.quadraturepoints(b1)) ≈ HelFEM.basisvalues(b1)
    b2(HelFEM.quadraturepoints(b2)) ≈ HelFEM.basisvalues(b2)

    # Each element has the same number of basis functions as the order of the polynomials
    # (nnodes), but then we merge the ones on the element boundaries together and also
    # remove one function from each edge to satisfy boundary conditions
    nbf(nnodes, nelems) = nnodes * nelems - (nelems - 1) - 2
    @test length(b1) == nbf(16, 10)
    @test length(b2) == nbf(6, 170)

    # Introspection for element boundaries
    @test_throws Exception HelFEM.elementrange(b1, 0)
    @test_throws Exception HelFEM.elementrange(b1, 11)
    @test HelFEM.elementrange(b1, 1)[1] == 0.0
    @test HelFEM.elementrange(b2, 1)[1] == 0.0
    @test HelFEM.elementrange(b1, 10)[2] == 40.0
    @test HelFEM.elementrange(b2, 170)[2] == 50.0

    let bs = HelFEM.boundaries(b1)
        @test length(bs) == 11
        @test bs[1] == 0.0
        @test bs[end] == 40.0
    end
    let bs = HelFEM.boundaries(b2)
        @test length(bs) == 171
        @test bs[1] == 0.0
        @test bs[end] == 50.0
    end

    # Scaling between polynomial coordinates to radial coordinates
    @test_throws Exception HelFEM.scale_to_element(b1, 0, 0)
    @test_throws Exception HelFEM.scale_to_element(b1, 11, 0)
    let rs = HelFEM.scale_to_element(b1, 1, -1)
        @test isa(rs, Number)
        @test rs == 0.0
    end
    let rs = HelFEM.scale_to_element(b1, 10, 1)
        @test isa(rs, Number)
        @test rs == 40.0
    end
    let rs = HelFEM.scale_to_element(b1, 1, [-1, 0, 1])
        @test length(rs) == 3
        @test rs[1] == 0.0
    end
    let rs = HelFEM.scale_to_element(b2, 170, [1, -0.5, 0, 0.5, 1])
        @test length(rs) == 5
        @test rs[1] == 50.0
        @test rs[5] == 50.0
        @test rs[2]+rs[4] ≈ 2*rs[3]
    end

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

    @test HelFEM.add_boundary!(b1, 1.23) === nothing
    @test length(HelFEM.boundaries(b1)) == 12
end
