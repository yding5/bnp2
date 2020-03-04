module BNP2

using PyCall, LinearAlgebra, ArgCheck, Reexport, FileIO, Images, LabelledArrays
@reexport using Parameters, MLToolkit.Plots
using PhysicalConstants: CODATA2014, CODATA2018

const G = CODATA2018.NewtonianConstantOfGravitation.val
const g = CODATA2014.StandardAccelerationOfGravitation.val
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

include("laws.jl")
include("world.jl")
export AbstractObject, massof, positionof, velocityof, Particle, Bar
export AbstractForce, Force, forceof
export AbstractEnvironment, stateof, accelerationof, EarthWithForce, EarthWithObjects, Space
include("simulators.jl")
export AbstractSimulator, simulate, transition, SimpleSimulator, DiffEqSimulator, PymunkSimulator, DynSysSimulator
include("vis.jl")
export animof, preview_frames, plot_force!

end # module
