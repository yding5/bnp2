using DrWatson
@quickactivate "BNP2"

args = (
    seed        = 1110,
    dataset     = "rand_init_three_body",
    n_train     = 400,
    n_valid     = 40,
    n_test      = 40,
    T           = 50,
    dt          = 1e-3,
    is_noisyobs = false,
    σ_obs       = 1e-1,
)

using BNP2, Random, BSON
include(srcdir("Data.jl"))
using .Data

savedir = datadir("diffeq-$(args.dataset)")
isdir(savedir) || mkdir(savedir)    # create directory if not exist

# Actual observation noise level
σ_obs = args.is_noisyobs * args.σ_obs

data_cfg = 
if args.dataset == "rand_init_three_body"
    Data.RandInitThreeBodyCfg(;
        ms = fill(5e10, 3),
        Q = cat(
            [-1,  0], 
            [ 1,  0], 
            [ 0, √3]; dims=2
        ),
        P = cat(
            [cos(π / 3), -sin(π / 3)], 
            [cos(π / 3),  sin(π / 3)], 
            [cos(π),      sin(π)]; dims=2
        ),
        n_train = args.n_train, 
        n_valid = args.n_valid, 
        n_test  = args.n_test, 
        T  = args.T,
        dt = args.dt, 
    )
end

trajs = Data.sim(MersenneTwister(args.seed), data_cfg)

@info "Simulated data" length(trajs.train) length(trajs.valid) length(trajs.test)

for (i, traj) in enumerate(trajs.test)
    savepath = joinpath(savedir, "test-$i.mp4")
    animof(traj).save(savepath)
end

data = (
    train = Data.preprocess(trajs.train; σ_obs=σ_obs), 
    valid = Data.preprocess(trajs.valid; σ_obs=σ_obs),
    test  = Data.preprocess(trajs.test;  σ_obs=σ_obs),
)

@info "Processed data" size(data.train) size(data.valid) size(data.test)

bson(joinpath(savedir, "data.bson"), trajs=trajs, data=data)

@info "Saved data into $savedir"