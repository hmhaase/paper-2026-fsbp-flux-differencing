# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

include("containers.jl")
include("containers_2d.jl")
include("fsbp_1d.jl")
include("fsbp_2d.jl")
end # @muladd
