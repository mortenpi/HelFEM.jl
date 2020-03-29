using HelFEM
using Test

@testset "HelFEM.jl" begin
    using HelFEM
    b = HelFEM.basis(10, 20, 4, 40, 2, 2.0, 0)
    m = HelFEM.radial_integral(b, b, 0, false, false)
    @test size(m, 1) === 179
    @test size(m, 2) === 179
    @test size(m, 3) === 1
    @test size(m) == (179, 179)
    @test_throws ArgumentError size(m, 0)
    @test_throws ArgumentError size(m, -100)
end
