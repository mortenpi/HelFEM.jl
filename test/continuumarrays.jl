# This file tests the ContinuumArrays-based API in HelFEM.CompactFEMBasis
using Test
using HelFEM: PolynomialBasis, FEMBasis
using HelFEM.CompactFEMBasis: HelFEMBasis
using HelFEM.CompactFEMBasis.QuasiArrays: Inclusion, domain, QuasiDiagonal
using HelFEM.CompactFEMBasis.ContinuumArrays: Derivative
@testset "CompactFEMBasis" begin
    pb = PolynomialBasis(:lip, 5)
    b = FEMBasis(pb, [-1, 1, 2])
    B = HelFEMBasis(b)

    r = axes(B, 1)
    r isa Inclusion
    @test minimum(domain(r)) == -1
    @test maximum(domain(r)) == 2
    @test length(axes(B, 2)) == length(b)

    let S = B'B
        @test S isa AbstractMatrix
        @test size(S) == (length(b), length(b))
    end
    let D = Derivative(r)
        BDB = B'D*B
        @test BDB isa AbstractMatrix
        @test size(BDB) == (length(b), length(b))
        BDDB = B'D*B
        @test BDDB isa AbstractMatrix
        @test size(BDDB) == (length(b), length(b))
    end
    let V = QuasiDiagonal((r -> r^2).(r))
        BVB = B'V*B
        @test BVB isa AbstractMatrix
        @test size(BVB) == (length(b), length(b))
    end
end
