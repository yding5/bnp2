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

using Random: AbstractRNG, GLOBAL_RNG

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

add_gaussiannoise(xs, sigma) = add_gaussiannoise(GLOBAL_RNG, xs, sigma)
function add_gaussiannoise(rng::AbstractRNG, xs::AbstractVector{<:AbstractVector}, sigma)
    return map(x -> x + sigma * randn(rng, size(x)...), xs)
end

export orthonormalvecof, rotate, add_gaussiannoise

### Phyics

using PhysicalConstants: CODATA2014, CODATA2018

const G = CODATA2018.NewtonianConstantOfGravitation.val
const g = CODATA2014.StandardAccelerationOfGravitation.val

function attractive_acceleration(m, r²)
    @argcheck !iszero(r²) "Attractive acceleration is undefined for 0 distance."
    return G * m / r²
end
attractive_force(m1, m2, r²) = m1 * attractive_acceleration(m2, r²)

include("world.jl")
include("simulators.jl")
export AbstractSimulator, simulate, transition, SimpleSimulator, DiffEqSimulator, PymunkSimulator
include("vis.jl")
export animof, preview_frames, plot_force!

end # module
