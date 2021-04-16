"""
CxxWrap wrapper module for the HelFEM shared library.
"""
module helfem
    using CxxWrap, HelFEM_jll
    @wrapmodule(libhelfem)
    function __init__()
        @initcxx
        verbose(false)
    end

    function ArmaVector(v::AbstractVector)
        av = ArmaVector(length(v))
        for (i, x) in enumerate(v)
            at!(av, i - 1, convert(Float64, x))
        end
        return av
    end
    Base.size(v::ArmaVector) = (Int(nrows(v)),)
    function Base.size(v::ArmaVector, dim::Integer)
        dim <= 0 && throw(ArgumentError("dimension $dim out of range"))
        dim == 1 ? Int(nrows(v)) : 1
    end
    Base.collect(v::ArmaVector) = [at(v, i - 1) for i = 1:size(v, 1)]

    function ArmaMatrix(M::AbstractMatrix)
        aM = ArmaMatrix(size(M, 1), size(M, 2))
        idxs = CartesianIndices(M)
        for (k, x) in enumerate(M)
            i, j = Tuple(idxs[k])
            at!(aM, i - 1, j - 1, convert(Float64, x))
        end
        return aM
    end
    Base.size(m::ArmaMatrix) = (Int(nrows(m)), Int(ncols(m)))
    function Base.size(m::ArmaMatrix, dim::Integer)
        dim <= 0 && throw(ArgumentError("dimension $dim out of range"))
        dim == 1 ? Int(nrows(m)) : dim == 2 ? Int(ncols(m)) : 1
    end
    Base.collect(m::ArmaMatrix) = [at(m, i - 1, j - 1) for i = 1:size(m, 1), j = 1:size(m, 2)]
end
