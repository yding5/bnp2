module BNP2

using PyCall

const pymunk = PyNULL()

__init__() = copy!(pymunk, pyimport("pymunk"))

include("world.jl")
export AbstractObject, Particle, massof, positionof, velocityof
export AbstractForce, Force, forceof
export AbstractEnvironment, Earth, getacceleration
include("simulators.jl")
export AbstractSimulator, simulate, SimpleSimulator, DiffEqSimulator, PymunkSimulator, transition

### Plots

using MLToolkit.Plots

function Plots.plot(traj::AbstractVector{T}, traj_ref=nothing) where {T<:Particle}
    pos, vel = hcat(positionof.(traj)...), hcat(velocityof.(traj)...)
    if !isnothing(traj_ref)
        pos_ref, vel_ref = hcat(positionof.(traj_ref)...), hcat(velocityof.(traj_ref)...)
    end

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(3 * 6, 4))
    let ax = axes[1]
        for i in 2:size(pos, 2)
            ax.plot([pos[1,i-1], pos[1,i]], [pos[2,i-1], pos[2,i]], "-")
        end
        for i in 2:size(pos, 2)
            ax.scatter([pos[1,i]], [pos[2,i]], s=16.0)
        end
        !isnothing(traj_ref) &&
            ax.plot(pos_ref[1,:], pos_ref[2,:], "--"; c="gray")
        ax.set_title("Position")
        ax.axis("equal")
    end
    let ax = axes[2]
        for i in 1:size(vel, 2)
            ax.plot([0, vel[1,i]], [0, vel[2,i]], "-o")
        end
        if !isnothing(traj_ref)
            for i in 1:size(vel, 2)
                ax.plot([0, vel_ref[1,i]], [0, vel_ref[2,i]], "--"; c="gray")
            end
        end
        ax.set_title("Velocity")
        ax.axis("equal")
    end

    let ax = axes[3]
        ax.plot(sqrt.(dropdims(sum(vel.^2; dims=1); dims=1)))
        !isnothing(traj_ref) && 
            ax.plot(sqrt.(dropdims(sum(vel_ref.^2; dims=1); dims=1)), "--"; c="gray")
        ax.set_title("Speed")
    end
    
    return fig, axes
end

export plot

end # module
