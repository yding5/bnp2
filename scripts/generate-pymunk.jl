using DrWatson
@quickactivate "BNP2"

args = (
    seed    = 1,
    dataset = "bouncing_ball",
    n_data  = 100,
)

using BNP2, Random, ProgressMeter

function generate_pymunk(obj, env, savedir, do_HTML=false)
    sim = PymunkSimulator(0.075)

    do_HTML && display(HTML(animof(simulate(obj, env, sim, 0)).to_html5_video()))
    animof(simulate(obj, env, sim, 0); savedir=savedir).save(joinpath(savedir, "frames.mp4"))
    symlink(joinpath(savedir, "frames.mp4"), "$savedir.mp4")
end

Random.seed!(args.seed)

@showprogress for i in 1:args.n_data
    savedir = datadir("pymunk-$(args.dataset)", "video-$i")
    obj = Particle(1.0, [0.0, 4.0] .+ randn(2), [2.5, 0.0] .+ randn(2))
    env = EarthWithObjects([
        Bar([-0.0, 0.0] .+ randn(2), [5.0, -0.0] .+ randn(2)), 
        Bar([5.0, -0.0] .+ randn(2), [5.0,  5.0] .+ randn(2)),
    ])
    generate_pymunk(obj, env, savedir)
end
