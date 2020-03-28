module HelFEM
using CxxWrap
@wrapmodule(joinpath(@__DIR__, "..", "deps", "lib", "libhelfem.so"))
function __init__()
    @initcxx
end
end # module
