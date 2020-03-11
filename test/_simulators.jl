@testset "simulators" begin
    D = 2
    mass = 2.0
    n = 3
    ms = fill(mass, n)
    S = randn(2, D, n)
    space = Space(Particle.(ms, S))
    earth = WithStatic(space, EARTH)
    
    dt = 1e-1
    n_steps = 10

    # Callbacks for SimpleSimulator
    env1 = transition(space, dt)
    envs = simulate(space, dt, n_steps)
    @test stateof(env1) == stateof(first(envs))
    
    for sim in [
        dt,
        SimpleSimulator(dt),
        DiffEqSimulator(dt),
    ], env in [
        space,
        earth,
    ]
        env1 = transition(env, sim)
        envs = simulate(env, sim, n_steps)
        @test stateof(env1) == stateof(first(envs))
    end

    let sim = PymunkSimulator(dt)
        for env in [
            space,
            earth,
            WithStatic(space, Bar(zeros(2), [0.0, 10.0])),
        ]
            env1 = transition(env, sim)
            envs = simulate(env, sim, n_steps)
            @test stateof(env1) == stateof(first(envs))
        end
    end
end
