@testset "world" begin
    D = 3
    mass = 2.0
    state = randn(2, D)
    pos = state[1,:]
    vel = state[2,:]
    n = 3
    ms = fill(mass, n)
    S = cat(state, randn(2, D, n - 1); dims=3)
    P = S[1,:,:]
    V = S[2,:,:]

    @testset "Particle" begin
        p = Particle(mass, state)
        @test p == objectof(p)
        @test mass == massof(p)
        @test state == stateof(p)
        @test pos == positionof(p)
        @test vel == velocityof(p)
        @test D == dimensionof(p)
        Particle(mass, D)
        Particle(mass, pos)
        Particle(mass, pos, nothing)
        Particle(mass, pos, vel)
        state′ = rand(size(state)...)
        p′ = reconstruct(p, state′)
        @test state′ == stateof(p′)
        ps = Particle.(ms, S)
        @test ms == massof.(ps)
        @test S == cat(stateof.(ps)...; dims=3)
        Particle.(ms, P)
        Particle.(ms, P, nothing)
        ps = Particle.(ms, P, V)
        @test ms == massof.(ps)
        @test S == cat(stateof.(ps)...; dims=3)
    end

    @testset "Forced" begin
        obj = Particle(mass, state)
        force = randn(D)
        f = Forced(obj, force)
        @test obj == objectof(f)
        @test mass == massof(f)
        @test state == stateof(f)
        @test pos == positionof(f)
        @test vel == velocityof(f)
        @test D == dimensionof(f)
        state′ = rand(size(state)...)
        f′ = reconstruct(f, state′)
        @test state′ == stateof(f′)
    end

    @testset "Bar" begin
        p1, p2 = [0, 0], [1, 0]
        thickness, elasticity = 0.3, 0.9
        Bar(p1, p2)
        Bar(p1, p2, thickness, elasticity)
    end

    @testset "GravitationalField" begin
        GravitationalField(rand(D), 1.0)
        EARTH
    end
    
    @testset "Space" begin
        obj = Particle(mass, state)
        space = Space(obj)
        @test tuple(obj) == objectsof(space)
        objs = Particle.(ms, S)
        space = Space(objs)
        @test S == stateof(space)
        @test P == positionof(space)
        @test V == velocityof(space)
        @test D == dimensionof(space)
        @test tuple(objs...) == objectsof(space)
        accelerationof(space)
        @test tuple() == staticof(space)
        foreach(1:n) do i
            accelerationof(space, i)
        end
        S′ = randn(size(S)...)
        space′ = reconstruct(space, S′)
        @test S′ == stateof(space′)
        P′ = S′[1,:,:]
        V′ = S′[2,:,:]
        space′ = reconstruct(space, P′)
        @test P′ == positionof(space′)
        space′ = reconstruct(space, P′, nothing)
        @test P′ == positionof(space′)
        space′ = reconstruct(space, P′, V′)
        @test P′ == positionof(space′)
        @test V′ == velocityof(space′)
    end

    @testset "WithStatic" begin
        objs = tuple(Particle.(ms, S)...)
        env = Space(objs)
        static = Bar([0, 0], [1, 0])
        w = WithStatic(env, static)
        @test env == envof(w)
        @test objs == objectsof(env)
        accelerationof(w)
        @test tuple(static) == staticof(w)
    end
end
