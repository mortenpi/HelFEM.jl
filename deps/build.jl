using CxxWrap
using libxc_jll
using armadillo_jll
using HDF5_jll
cxx_prefix = CxxWrap.CxxWrapCore.prefix_path()

libxc = joinpath(libxc_jll.artifact_dir, "lib", "libxc.a")
@assert isfile(libxc)

const helfem_commit = "e9ab305933595d48f3badc72431c3fe46eb4cfc8"

cd(@__DIR__) do
    for dir in ["libhelfem", "lib"]
        isdir(dir) || continue
        @warn "$dir/ exists, removing " pwd()
        rm(dir, recursive=true)
    end
    run(`git clone -n git@github.com:mortenpi/HelFEM-private.git libhelfem`)
    run(`git -C libhelfem/ checkout $(helfem_commit)`)

    libhelfem_src = joinpath(pwd(), "libhelfem", "src")
    @assert isdir(libhelfem_src)

    run(`cmake
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_PREFIX_PATH=$cxx_prefix
        -DLIBHELFEM_SRC=$libhelfem_src
        -DLIBXC=$(libxc_jll.artifact_dir)
        -DARMADILLO=$(armadillo_jll.artifact_dir)
        -DHDF5=$(HDF5_jll.artifact_dir)
        .`)
    run(`cmake --build . --config Release`)

    @assert isfile("lib/libhelfem.so")
end
