module HelFEM
using CxxWrap
import armadillo_jll, OpenBLAS_jll, HDF5_jll
@wrapmodule(joinpath(@__DIR__, "..", "deps", "lib", "libhelfem.so"))
function __init__()
    @initcxx
end

Base.size(m::ArmaMatrix) = (Int(nrows(m)), Int(ncols(m)))
function Base.size(m::ArmaMatrix, dim::Integer)
    dim <= 0 && throw(ArgumentError("dimension $dim out of range"))
    dim == 1 ? Int(nrows(m)) : dim == 2 ? Int(ncols(m)) : 1
end

end # module
