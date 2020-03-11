module BNP2

using PyCall, LinearAlgebra, Statistics, ArgCheck, Reexport, FileIO, Images
@reexport using Parameters, MLToolkit.Plots
import Parameters: reconstruct
import MLToolkit.Plots: plot!

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

include("utilites.jl")

include("world.jl")
export AbstractObject, Particle, Forced, Bar, GravitationalField, EARTH
export objectof, massof, stateof, positionof, velocityof, dimensionof
export AbstractEnvironment, WithStatic, Space
export envof, objectsof, accelerationof, staticof

include("simulators.jl")
export AbstractSimulator, simulate, transition, SimpleSimulator, DiffEqSimulator, PymunkSimulator

include("vis.jl")
export animof, preview_frames, plot_force!

end # module
