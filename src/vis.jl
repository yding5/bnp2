### Plots

function plot!(ax, p::Particle{2}; color=COLORS[:red], alpha=1.0, do_plotvelocity=false, kwargs...)
    position, velocity = positionof(p), velocityof(p)
    ax.scatter([position[1]], [position[2]]; color=color, alpha=alpha, linewidths=2.5)
    if do_plotvelocity && norm(velocity) > 0
        ax.arrow(
            position..., velocity...; 
            shape="left", width=arrowwidthof(ax), color=COLORS[:blue], alpha=0.5
        )
    end
end

function plot!(ax, bar::Bar; kwargs...)
    ax.plot(
        [bar.p1[1], bar.p2[1]], [bar.p1[2], bar.p2[2]]; 
        linewidth=20bar.thickness, color=COLORS[:gray]
    )
end

function arrowwidthof(ax)
    xlim = ax.get_xlim()
    Δx = abs(xlim[1] - xlim[2])
    return 1 / Δx / 200
end

function plot_force!(ax, pos, force)
    ax.arrow(pos..., force...; shape="left", width=arrowwidthof(ax), color=COLORS[:purple], alpha=0.5)
end

function plot!(ax, f::Forced; kwargs...)
    obj = objectof(f)
    plot!(ax, obj)
    pos = positionof(f)
    force = f.force
    plot_force!(ax, pos, force)
end

function plot!(ax, gf::GravitationalField; do_displayfiled=false, kwargs...)
    if do_displayfiled
        xlim = ax.get_xlim()
        xmid = mean(xlim)
        ylim = ax.get_ylim()
        ymid = mean(ylim)
        dy = abs(ylim[1] - ylim[2]) / 2
        u = gf.direction
        s = gf.strength / 100
        ax.arrow(xmid, ymid, u[1] * s, u[2] * s; width=arrowwidthof(ax), color=COLORS[:pink], alpha=0.5)
    end
end

function plot!(ax, env::AbstractEnvironment; kwargs...)
    for obj in objectsof(env)
        plot!(ax, obj; kwargs...)
    end
    for s in staticof(env)
        plot!(ax, s; kwargs...)
    end
end

function plot!(ax, traj::AbstractVector{<:AbstractEnvironment}; alpha=0.5, kwargs...)
    Q = cat(positionof.(traj)...; dims=3)
    for i in 1:size(Q, 2)
        plot!(ax, TwoDimPath(Q[1,i,:], Q[2,i,:]), "-"; alpha=alpha)
    end
end

### Animation

function animof(env::AbstractEnvironment, sim::AbstractSimulator, T::Int; kwargs...)
    traj = simulate(env, sim, T)
    return animof(traj; kwargs...)
end

function animof(
    traj::AbstractVector{<:AbstractEnvironment}, traj_ref=nothing; 
    d=2, do_tracklocal=false, do_plotpath=true, do_plotinit=true, refs=[], kwargs...
)
    @argcheck dimensionof(first(traj)) == 2
    
    Q = cat(positionof.(traj)...; dims=3)
    Q_ref = isnothing(traj_ref) ? nothing : cat(positionof.(traj_ref)...; dims=3)

    dev = max(3maximum(std(Q[1,:,:])), 3maximum(std(Q[2,:,:])))
    xmid_global, ymid_global = mean(Q[1,:,:]), mean(Q[2,:,:])

    fig, ax = figure(figsize=(5, 5))
    
    init!() = ax.clear()
    
    function draw!(t)
        ax.clear()
        plot!(ax, traj[t])
        if do_tracklocal
            xs, ys = Q[1,:,t], Q[2,:,t]
            xmid, ymid = mean(xs), mean(ys)
        else
            xmid, ymid = xmid_global, ymid_global
        end
        ax.set_xlim([xmid - dev, xmid + dev])
        ax.set_ylim([ymid - dev, ymid + dev])
        do_plotpath && for i in 1:size(Q, 2)
            plot!(ax, TwoDimPath(Q[1,i,1:t], Q[2,i,1:t]), "-"; c=COLORS[:gray], alpha=0.5)
        end
        do_plotinit && plot!(ax, traj[1]; color=COLORS[:gray], alpha=0.5)
        !isnothing(Q_ref) && for i in 1:size(Q_ref, 2)
            plot!(ax, TwoDimPath(Q_ref[1,i,:], Q_ref[2,i,:]), "--"; c=COLORS[:pink], alpha=0.5)
        end
    end
    
    return animation.FuncAnimation(fig, draw!; init_func=init!, frames=1:length(traj), interval=50, blit=false)
end

###

# function plot(traj::AbstractVector{T}, traj_ref=nothing; do_scatter=false) where {T<:Particle}
#     pos, vel = hcat(positionof.(traj)...), hcat(velocityof.(traj)...)
#     if !isnothing(traj_ref)
#         pos_ref, vel_ref = hcat(positionof.(traj_ref)...), hcat(velocityof.(traj_ref)...)
#     end

#     cmap = cm.ScalarMappable(norm=colors.Normalize(0, 1), cmap="Greys")

#     fig, axes = figure(1, 3; figsize=(3 * 6, 4))
#     let ax = axes[1]
#         len = size(pos, 2)
#         for i in 2:len
#             c = cmap.to_rgba(i / len)
#             ax.plot([pos[1,i-1], pos[1,i]], [pos[2,i-1], pos[2,i]], "-", c=c)
#             do_scatter && ax.scatter([pos[1,i]], [pos[2,i]], s=16.0, c=[c])
#         end
#         !isnothing(traj_ref) &&
#             ax.plot(pos_ref[1,:], pos_ref[2,:], "--"; c="gray")
#         ax.set_title("Position")
#         ax.axis("equal")
#     end
#     let ax = axes[2]
#         len = size(vel, 2)
#         for i in 1:len
#             ax.plot([0, vel[1,i]], [0, vel[2,i]], "-", c=cmap.to_rgba(i / len))
#         end
#         if !isnothing(traj_ref)
#             for i in 1:size(vel_ref, 2)
#                 ax.plot([0, vel_ref[1,i]], [0, vel_ref[2,i]], "--"; c="gray")
#             end
#         end
#         ax.set_title("Velocity")
#         ax.axis("equal")
#     end

#     let ax = axes[3]
#         ax.plot(sqrt.(dropdims(sum(vel.^2; dims=1); dims=1)))
#         !isnothing(traj_ref) && 
#             ax.plot(sqrt.(dropdims(sum(vel_ref.^2; dims=1); dims=1)), "--"; c="gray")
#         ax.set_title("Speed")
#     end
    
#     return fig, axes
# end

# function preview_frames(videodir, frame_idcs)
#     n = length(frame_idcs)
#     @assert n > 1 "Only support previewing more than 1 frames."
#     img = nothing
#     fig, axes = figure(1, n; figsize=(3 * n, 3))
#     for (ax, id) in zip(axes, frame_idcs)
#         img = load(joinpath(videodir, "frame-$id.png"))
#         img = Array(channelview(float.(img)))
#         img = permutedims(img, (2, 3, 1))
#         img = img[:,:,1:3]
#         ax.imshow(img)
#         ax.set_title("frame-$id")
#         ax.set_xticks([])
#         ax.set_yticks([])
#     end
#     display(fig)
#     println("size(img): ", size(img))
# end
