using DrWatson
@quickactivate "BNP2"
using Test, BNP2

@testset "AbstractObject" begin
    @testset "Particle" begin
        mass = 1.0
        dim = 2
        state = randn(2, dim)
        position = state[1,:]
        velocity = state[2,:]
        p = Particle(mass, state)
        @test mass == massof(p)
        @test state == stateof(p)
        @test position == positionof(p)
        @test velocity == velocityof(p)
        @test dim == dimensionof(p)
        Particle(mass, dim)
        Particle(mass, position)
        Particle(mass, position, nothing)
        Particle(mass, position, velocity)
        state′ = rand(2, dim)
        p′ = reconstruct(p, state′)
        @test state′ == stateof(p′)
        n = 3
        Particle.(fill(mass, n), repeat(state, 1, n))
        Particle.(fill(mass, n), repeat(position, n))
        Particle.(fill(mass, n), repeat(position, n), nothing)
        Particle.(fill(mass, n), repeat(position, n), repeat(velocity, n))
    end
end
