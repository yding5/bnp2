### Object

abstract type AbstractObject end

"""Particle"""
struct Particle{D, M, S} <: AbstractObject
    mass::M     # (kg)
    state::S    # position (m) & velocity (m^2 s^-1)
end

massof(p::Particle) = p.mass
stateof(p::Particle) = p.state
positionof(p::Particle) = p.state[1,:]
velocityof(p::Particle) = p.state[2,:]
dimensionof(p::Particle{D}) where {D} = D

Particle(m::M, s::S) where {M, S} = Particle{size(s, 2), M, S}(m, s)
Particle(mass::T, dim::Int) where {T} = Particle(mass, zeros(T, 2, dim))
Particle(mass, p::AbstractVector{T}) where {T} = Particle(mass, p, zeros(T, length(p)))
Particle(mass, p::AbstractVector, ::Nothing) = Particle(mass, p)
Particle(mass, p::T, v::T) where {T<:AbstractVector} = Particle(mass, vcat(p', v'))
reconstruct(p::Particle, args...) = Particle(p.mass, args...)

function Base.Broadcast.broadcasted(
    ::typeof(Particle), ms::AbstractVector, states::AbstractMatrix
)
    return Particle.(ms, _tolist(div(size(states, 2), length(ms)), states))
end

function Base.Broadcast.broadcasted(
    ::typeof(Particle), ms::AbstractVector, ps::AbstractVector{<:Real}, vs=nothing
)
    return Particle.(ms, _tolist.(div(length(ps), length(ms)), (ps, vs))...)
end

function forceof(p1::Particle, p2::Particle)
    p1 === p2 && return nothing
    Δp = positionof(p2) - positionof(p1)
    r² = sum(abs2, Δp)
    u = Δp / sqrt(r²)
    return attractive_force(p1.mass, p2.mass, r²) * u
end

"""Bar"""
struct Bar{T1, T2} <: AbstractObject
    pstart::T1
    pend::T1
    tickness::T2
    elasticity::T2
end

Bar(pstart, pend) = Bar(pstart, pend, 0.3, 0.9)

# """GravitationalField"""
# struct GravitationalField{S} <: AbstractObject
#     strength::S
# end

### Environment

abstract type AbstractEnvironment end

stateof(env::AbstractEnvironment) = hcat(stateof.(objectsof(env))...)
positionof(env::AbstractEnvironment) = vcat(positionof.(objectsof(env))...)
velocityof(env::AbstractEnvironment) = vcat(velocityof.(objectsof(env))...)
dimensionof(env::AbstractEnvironment) = dimensionof(first(objectsof(env)))
function accelerationof(env::AbstractEnvironment, i::Int)
    d = dimensionof(env)
    return accelerationof(env)[(i-1)*d+1:i*d]
end

abstract type AbstractWithEnvironment <: AbstractEnvironment end

envof(w::AbstractWithEnvironment) = w.env
objectsof(w::AbstractWithEnvironment) = objectsof(envof(w))
accelerationof(w::AbstractWithEnvironment) = accelerationof(envof(w)) + w._cache

"""WithForce"""
struct WithForce{
    E<:AbstractEnvironment, F<:Dict{Int, <:AbstractVector}, C
} <: AbstractWithEnvironment
    env::E
    forces::F
    _cache::C
end

forceof(wf::WithForce) = wf.forces

WithForce(env, forces::Pair...) = WithForce(env, Dict(forces...))
function WithForce(env, forces::Dict)
    _cache = acceleration_by_external(objectsof(env), forces)
    return WithForce(env, forces, _cache)
end

function acceleration_by_external(objects, forces)
    as = []
    for (i, o) in enumerate(objects)
        a = i in keys(forces) ? forces[i] / massof(o) : zeros(dimensionof(o))
        push!(as, a)
    end
    return vcat(as...)
end

reconstruct(wf::WithForce, args...) = WithForce(reconstruct(envof(wf), args...), wf.forces)

"""WithStatic"""
struct WithStatic{E<:AbstractEnvironment, S, C} <: AbstractWithEnvironment
    env::E
    static::S
    _cache::C
end

function WithStatic(env, static)
    _cache = zeros(length(objectsof(env)) * dimensionof(env))
    return WithStatic(env, static, _cache)
end

staticof(ws::WithStatic) = ws.static

reconstruct(ws::WithStatic, args...) = WithStatic(reconstruct(envof(ws), args...), ws.static)

"""Space"""
struct Space{O<:Tuple, C} <: AbstractEnvironment
    objects::O
    _cache::C
end

objectsof(s::Space) = s.objects
accelerationof(s::Space) = s._cache

Space(o::AbstractObject, args...) = Space(tuple(o), args...)
Space(os::AbstractVector, args...) = Space(tuple(os...), args...)
function Space(os::Tuple)
    @argcheck all(dimensionof(first(os)) .== dimensionof.(os))
    _cache = acceleration_by_interaction(os)
    return Space(os, _cache)
end

acceleration_by_interaction(os::NTuple{1}) = zeros(dimensionof(first(os)))
function acceleration_by_interaction(objects::Tuple)
    as = []
    for o1 in objects
        objects′ = filter(o2 -> !(o1 === o2), objects)
        if length(objects′) > 0
            f1 = mapreduce(o2 -> forceof(o1, o2), +, objects′)
        else
            f1 = zeros(dimensionof(o1))
        end
        push!(as, f1 / massof(o1))
    end
    return vcat(as...)
end

reconstruct(s::Space, objects) = Space(objects)
reconstruct(s::Space, states::AbstractMatrix) = 
    Space(reconstruct.(objectsof(s), _tolist(dimensionof(s), states)))
reconstruct(s::Space, ps::AbstractVector{<:Real}, vs=nothing) = 
    Space(reconstruct.(objectsof(s), _tolist.(dimensionof(s), (ps, vs))...))
