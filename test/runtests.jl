using DrWatson
@quickactivate "BNP2"
using Test, BNP2

@testset "AbstractObject" begin
    dim = 2
    mass = 1.0
    state = randn(2, dim)
    position = state[1,:]
    velocity = state[2,:]
    n = 3
    ms = fill(mass, n)
    states = repeat(state, 1, n)
    ps = states[1,:]
    vs = states[2,:]
    @testset "Particle" begin
        p = Particle(mass, state)
        @test mass == massof(p)
        @test state == stateof(p)
        @test position == positionof(p)
        @test velocity == velocityof(p)
        @test dim == dimensionof(p)
        @test nothing == forceof(p)
        Particle(mass, dim)
        Particle(mass, position)
        Particle(mass, position, nothing)
        Particle(mass, position, velocity)
        state′ = rand(size(state)...)
        p′ = reconstruct(p, state′)
        @test state′ == stateof(p′)
        Particle.(ms, states)
        Particle.(ms, ps)
        Particle.(ms, ps, nothing)
        Particle.(ms, ps, vs)
    end
    @testset "Forced" begin
        p = Particle(mass, state)
        f = randn(dim)
        fp = Forced(p, f)
        @test mass == massof(fp)
        @test state == stateof(fp)
        @test position == positionof(fp)
        @test velocity == velocityof(fp)
        @test dim == dimensionof(fp)
        @test f == forceof(fp)
        fps = Forced.(Particle.(ms, states), Ref(f))
    end
    @testset "Space" begin
        objs =  Particle.(ms, states)
        space = Space(objs)
        @test states == stateof(space)
        @test ps == positionof(space)
        @test vs == velocityof(space)
        @test dim == dimensionof(space)
        @test tuple(objs...) == objectsof(space)
        accelerationof(space)
        states′ = rand(size(states)...)
        space′ = reconstruct(space, states′)
        @test states′ == stateof(space′)
        ps′ = states′[1,:]
        vs′ = states′[2,:]
        space′ = reconstruct(space, ps′)
        @test ps′ == positionof(space′)
        space′ = reconstruct(space, ps′, nothing)
        @test ps′ == positionof(space′)
        space′ = reconstruct(space, ps′, vs′)
        @test ps′ == positionof(space′)
        @test vs′ == velocityof(space′)
    end
end
