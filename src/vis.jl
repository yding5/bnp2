### Plots

function Plots.plot(traj::AbstractVector{T}, traj_ref=nothing; do_scatter=false) where {T<:Particle}
    pos, vel = hcat(positionof.(traj)...), hcat(velocityof.(traj)...)
    if !isnothing(traj_ref)
        pos_ref, vel_ref = hcat(positionof.(traj_ref)...), hcat(velocityof.(traj_ref)...)
    end

    cmap = cm.ScalarMappable(norm=colors.Normalize(0, 1), cmap="Greys")

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(3 * 6, 4))
    let ax = axes[1]
        len = size(pos, 2)
        for i in 2:len
            c = cmap.to_rgba(1 - i / len)
            ax.plot([pos[1,i-1], pos[1,i]], [pos[2,i-1], pos[2,i]], "-", c=c)
            do_scatter && ax.scatter([pos[1,i]], [pos[2,i]], s=16.0, c=[c])
        end
        !isnothing(traj_ref) &&
            ax.plot(pos_ref[1,:], pos_ref[2,:], "--"; c="gray")
        ax.set_title("Position")
        ax.axis("equal")
    end
    let ax = axes[2]
        len = size(vel, 2)
        for i in 1:len
            ax.plot([0, vel[1,i]], [0, vel[2,i]], "-", c=cmap.to_rgba(1 - i / len))
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

function arrowwidthof(ax)
    xlim = ax.get_xlim()
    Δx = abs(xlim[1] - xlim[2])
    return 1 / Δx / 5
end

function Plots.plot!(ax, p::Particle{2}; do_plotvelocity=false)
    @unpack position, velocity = p
    ax.scatter([position[1]], [position[2]], color="#E24A33")
    if do_plotvelocity && norm(velocity) > 0
        ax.arrow(
            position..., velocity...; 
            shape="left", width=arrowwidthof(ax), color="#348ABD", alpha=0.5
        )
    end
end

function plot_force!(ax, p, f)
    f = f / 0.1e12 # trillion Newton
    ax.arrow(
        p.position..., f...; 
        shape="left", width=arrowwidthof(ax), color="#988ED5", alpha=0.5
    )
end

function preview_frames(videodir, frame_idcs)
    n = length(frame_idcs)
    @assert n > 1 "Only support previewing more than 1 frames."
    img = nothing
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    for (ax, id) in zip(axes, frame_idcs)
        img = load(joinpath(videodir, "frame-$id.png"))
        img = Array(channelview(float.(img)))
        img = permutedims(img, (2, 3, 1))
        img = img[:,:,1:3]
        ax.imshow(img)
        ax.set_title("frame-$id")
        ax.set_xticks([])
        ax.set_yticks([])
    end
    display(fig)
    println("size(img): ", size(img))
end

### Animation

function init_anim(space, sim::PymunkSimulator; savedir=nothing)
    @unpack dt = sim
    fig = plt.figure(figsize=(3, 3))
    ax = plt.axes()
    options = matplotlib_util.DrawOptions(ax)
    options.flags = pymunk.SpaceDebugDrawOptions.DRAW_SHAPES
    options.flags |= pymunk.SpaceDebugDrawOptions.DRAW_CONSTRAINTS
    counter = Ref(1)
    function reset_ax!()
        ax.set_xlim((-2.5, 7.5))
        ax.set_ylim((-2.5, 7.5))
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    end
    function init_func()
        reset_ax!()
        space.debug_draw(options)
        return []
    end
    function func(_dt)
        for x in 1:10
            space.step(dt / 10)
        end
        ax.clear()
        reset_ax!()
        space.debug_draw(options)
        if !isnothing(savedir)
            !ispath(savedir) && mkpath(savedir)
            fig.savefig(joinpath(savedir, "frame-$(counter[]).png"); bbox_inches="tight", pad_inches=0, dpi=52.5)
        end
        counter[] += 1
        return []
    end
    return fig, init_func, func
end

animof(t::Tuple; kwargs...) = animof(t...; kwargs...)
function animof(space, sim; kwargs...)
    fig, init_func, func = init_anim(space, sim; kwargs...)
    return animation.FuncAnimation(fig, func, init_func=init_func, frames=64, interval=35, blit=false)
end

function animof(env, sim::DiffEqSimulator, n_frames; xlim=nothing, ylim=nothing, kwargs...)
    traj = simulate(env, sim, n_frames)
    ps = positionof.(traj)
    vs = velocityof.(traj)
    return animof(hcat(ps...), hcat(vs...); xlim=xlim, ylim=ylim, kwargs...)
end

function animof(Q, P=nothing; xlim=nothing, ylim=nothing, kwargs...)
    fig, ax = plt.subplots(figsize=(5, 5))
    
    function reset!(ax)
        ax.clear()
        !isnothing(xlim) && ax.set_xlim(xlim)
        !isnothing(ylim) && ax.set_ylim(ylim)
    end
    
    init!() = reset!(ax)
    
    function draw!(t)
        reset!(ax)
        for i in 1:3
            plot!(ax, TwoDimPath(Q[2i-1,1:t], Q[2i,1:t]), "--"; c="#777777", alpha=0.5)
        end
        plot!.(Ref(ax), 
            Particle.(nothing, _tolist(Q[:,1]), isnothing(P) ? nothing : _tolist(P[:,1]))
        )
        plot!.(Ref(ax), 
            Particle.(nothing, _tolist(Q[:,t]), isnothing(P) ? nothing : _tolist(P[:,t])); 
            do_plotvelocity=true
        )
    end
    
    anim = animation.FuncAnimation(fig, draw!; init_func=init!, frames=1:size(Q, 2), interval=50, blit=false)
end
