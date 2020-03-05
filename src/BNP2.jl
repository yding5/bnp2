module BNP2

using PyCall, LinearAlgebra, Statistics, ArgCheck, Reexport, FileIO, Images, LabelledArrays
@reexport using Parameters, MLToolkit.Plots
using PhysicalConstants: CODATA2014, CODATA2018
import Parameters: reconstruct

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

_tolist(d::Int, v::AbstractVector) = [v[i:i+d-1] for i in 1:d:length(v)]
_tolist(d::Int, m::AbstractMatrix) = [m[:,i:i+d-1] for i in 1:d:size(m, 2)]
_tolist(d::Int, ::Nothing) = nothing

function orthonormalvecof(v::AbstractVector)
    @argcheck length(v) == 2
    return [v[2], -v[1]] / sqrt(sum(v.^2))
end

include("laws.jl")
include("world.jl")
export AbstractObject, massof, stateof, positionof, velocityof, dimensionof, forceof
export Forced, Particle, Bar
export AbstractEnvironment, objectsof, accelerationof
export Space, WithStatic
include("simulators.jl")
export AbstractSimulator, simulate, transition, SimpleSimulator, DiffEqSimulator, PymunkSimulator
include("vis.jl")
export animof, preview_frames, plot_force!

end # module
