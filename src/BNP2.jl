module BNP2

using PyCall, LinearAlgebra, Statistics, ArgCheck, Reexport, FileIO, Images, LabelledArrays
@reexport using Parameters, MLToolkit.Plots

const cm = PyNULL()
const colors = PyNULL()
const pymunk = PyNULL()
const animation = PyNULL()
const matplotlib_util = PyNULL()

function __init__()
    copy!(cm, pyimport("matplotlib.cm"))
    copy!(colors, pyimport("matplotlib.colors"))
    copy!(pymunk, pyimport("pymunk"))
    copy!(animation, pyimport("matplotlib.animation"))
    copy!(matplotlib_util, pyimport("pymunk.matplotlib_util"))
end

### Utilites

function orthonormalvecof(v::AbstractVector)
    @argcheck length(v) == 2
    return [v[2], -v[1]] / sqrt(sum(v.^2))
end

function rotate(θ, x)
    # Rotation matrix
    R = [
        cos(θ) -sin(θ); 
        sin(θ)  cos(θ)
    ]
    return R * x
end

export orthonormalvecof, rotate

function apply_kernels(X)
    return vcat(X, 1 ./ X, sin.(X), cos.(X))
end

function euclidsq(X::T) where {T<:AbstractMatrix}
    XiXj = transpose(X) * X
    x² = sum(X.^2; dims=1)
    return transpose(x²) .+ x² - 2XiXj
end

function pairwise_compute(X)
    dim = div(size(X, 1), 3)
    X = cat([X[(i-1)*dim+1:i*dim,:] for i in 1:3]...; dims=3)
    hs = map(1:size(X, 2)) do t
        Xt = X[:,t,:]
        Dt = euclidsq(Xt)
        ht = sum(Dt; dims=2)
    end
    return hcat(hs...)
end

include("World.jl")
@reexport using .World
include("simulators.jl")
export AbstractSimulator, simulate, transition, SimpleSimulator, DiffEqSimulator, PymunkSimulator
include("vis.jl")
export animof, preview_frames, plot_force!

end # module
