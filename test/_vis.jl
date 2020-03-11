@testset "vis" begin
    @testset "plot!" begin
        n_figs, fs = 5, 4
        
        fig, axes = figure(1, n_figs; figsize=(n_figs * fs, fs))
        p = Particle(1.0, rand(2), rand(2))
        let ax = axes[1]
            plot!(ax, p)
        end
        n_objs = 3
        env = Space(Particle.(rand(n_objs), rand(2, n_objs), rand(2, n_objs)))
        let ax = axes[2]
            plot!(ax, env)
        end
        let ax = axes[3]
            objs = objectsof(env)
            plot!(ax, Space((objs[1:2]..., Forced(objs[3], rand(2) / 10))))
        end
        let ax = axes[4]
            plot!(ax, WithStatic(env, Bar([0, 0], [1, 0])))
        end
        let ax = axes[5]
            plot!(ax, WithStatic(env, EARTH))
        end

        savefig(fig, "vis.png")
    end

    @testset "animof" begin
        ms = fill(5e10, 3)
        qs = [[-1, 0], [1, 0], [0, √3]]
        ps = [[cos(π/3), -sin(π/3)], [cos(π/3),  sin(π/3)], [cos(π),  sin(π)]] .+ [rand(2) / 2]
        
        dt, T = 1e-1, 50
        sim = DiffEqSimulator(dt)
        traj = simulate(Space(Particle.(ms, qs, ps)), sim, T)
        traj_ref = simulate(Space(Particle.(ms, qs, ps .+ [rand(2) / 10])), sim, T)
        
        for (i, anim) in enumerate([
            animof(traj; do_plotpath=false, do_plotinit=false),
            animof(traj; do_tracklocal=true),
            animof(traj),
            animof(traj, traj_ref; do_plotpath=true, do_plotinit=true)
        ])
            anim.save("anim-$i.mp4")
        end

        obj0 = Particle(1.0, [0.0, 2.5], [2.5, 0.0])
        bars = [Bar([-5.0, 0.0], [5.0, 0.0]), Bar([5.5, 0.0], [5.5, 5.0])]
        env = WithStatic(Space(obj0), (EARTH, bars...))
        animof(env, PymunkSimulator(0.05), 100).save("anim-bouncing_ball.mp4")
    end
end
