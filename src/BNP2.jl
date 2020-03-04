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

_tolist(d::Int, v) = [v[i:i+d-1] for i in 1:d:length(v)]
_tolist(d::Int, ::Nothing) = nothing

include("laws.jl")
include("world.jl")
export AbstractObject, massof, positionof, velocityof, Particle, Bar
export AbstractForce, Force, forceof
export AbstractEnvironment, stateof, accelerationof, Space, Earth
include("simulators.jl")
export AbstractSimulator, simulate, transition, SimpleSimulator, DiffEqSimulator, PymunkSimulator
include("vis.jl")
export animof, preview_frames, plot_force!

end # module
