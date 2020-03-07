### Object

abstract type AbstractObject end

objectof(obj::AbstractObject) = obj
forceof(obj1, obj2) = massof(obj1) * accelerationof(obj1, obj2)

function accelerationof(obj1::AbstractObject, obj2::AbstractObject)
    obj1 === obj2 && return nothing
    Δp = positionof(obj2) - positionof(obj1)
    r² = sum(abs2, Δp)
    u = Δp / sqrt(r²)
    return attractive_acceleration(massof(obj2), r²) * u
end

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

function Base.Broadcast.broadcasted(::typeof(Particle), ms::AbstractVector, states::AbstractMatrix)
    return Particle.(ms, _tolist(div(size(states, 2), length(ms)), states))
end

function Base.Broadcast.broadcasted(::typeof(Particle), ms::AbstractVector, ps::AbstractVector{<:Real}, vs=nothing)
    return Particle.(ms, _tolist.(div(length(ps), length(ms)), (ps, vs))...)
end

"""Forced"""
struct Forced{O<:AbstractObject, F} <: AbstractObject
    obj::O
    force::F
end

objectof(f::Forced) = f.obj
massof(f::Forced) = massof(f.obj)
stateof(f::Forced) = stateof(f.obj)
positionof(f::Forced) = positionof(f.obj)
velocityof(f::Forced) = velocityof(f.obj)
dimensionof(f::Forced) = dimensionof(f.obj)

reconstruct(f::Forced, args...) = Forced(reconstruct(f.obj, args...), f.force)

accelerationof(f::Forced) = f.force / massof(f.obj)

"""Bar"""
struct Bar{T1, T2} <: AbstractObject
    pstart::T1
    pend::T1
    tickness::T2
    elasticity::T2
end

Bar(pstart, pend) = Bar(pstart, pend, 0.3, 0.9)

"""GravitationalField"""
struct GravitationalField{D, S} <: AbstractObject
    direction::D
    strength::S
end

function accelerationof(::AbstractObject, f::GravitationalField)
    u = f.direction / sqrt(sum(abs2, f.direction))
    return f.strength * u
end

const EARTH = GravitationalField([0, -1], g)

### Environment

abstract type AbstractEnvironment end

stateof(env::AbstractEnvironment) = hcat(positionof(env), velocityof(env))
positionof(env::AbstractEnvironment) = vcat(positionof.(objectsof(env))...)
velocityof(env::AbstractEnvironment) = vcat(velocityof.(objectsof(env))...)
dimensionof(env::AbstractEnvironment) = dimensionof(first(objectsof(env)))
function accelerationof(env::AbstractEnvironment, i::Int)
    d = dimensionof(env)
    return accelerationof(env)[(i-1)*d+1:i*d]
end
staticof(env::AbstractEnvironment) = staticof(envof(env))

"""WithStatic"""
struct WithStatic{E<:AbstractEnvironment, S, C} <: AbstractEnvironment
    env::E
    static::S
    _cache::C
end

envof(w::WithStatic) = w.env
objectsof(w::WithStatic) = objectsof(w.env)
accelerationof(w::WithStatic) = accelerationof(w.env) + w._cache
staticof(ws::WithStatic) = ws.static

WithStatic(env, static::AbstractObject) = WithStatic(env, tuple(static))
function WithStatic(env, static)
    as = map(objectsof(env)) do obj
        a = zeros(dimensionof(obj))
        for s in static
            if s isa GravitationalField
                a += accelerationof(obj, s)
            end
        end
        a
    end
    _cache = vcat(as...)
    return WithStatic(env, static, _cache)
end

reconstruct(ws::WithStatic, args...) = WithStatic(reconstruct(envof(ws), args...), ws.static)

"""Space"""
struct Space{O<:Tuple, C} <: AbstractEnvironment
    objects::O
    _cache::C
end

envof(s::Space) = s
objectsof(s::Space) = s.objects
accelerationof(s::Space) = s._cache
staticof(s::Space) = tuple()

Space(obj::AbstractObject, args...) = Space(tuple(obj), args...)
Space(objs::AbstractVector, args...) = Space(tuple(objs...), args...)
function Space(objs::Tuple)
    @argcheck all(dimensionof(first(objs)) .== dimensionof.(objs))
    _cache = acceleration_by_interaction(objs)
    return Space(objs, _cache)
end

acceleration_by_interaction(objs::NTuple{1,<:Particle}) = zeros(dimensionof(first(objs)))
acceleration_by_interaction(objs::NTuple{1,<:Forced}) = accelerationof(first(objs))
function acceleration_by_interaction(objects::Tuple)
    as = map(objects) do obj1
        objects′ = filter(obj2 -> !(obj1 === obj2), objects)
        a = zeros(dimensionof(obj1))
        if length(objects′) > 0
            for obj2 in objects′
                a += accelerationof(obj1, obj2)
            end
        end
        obj1 isa Forced ? a + accelerationof(obj1) : a
    end
    return vcat(as...)
end

reconstruct(s::Space, objects) = Space(objects)
reconstruct(s::Space, states::AbstractMatrix) = 
    Space(reconstruct.(objectsof(s), _tolist(dimensionof(s), states)))
function reconstruct(s::Space, pvec::AbstractVector{<:Real}, vvec::AbstractVector{<:Real})
    objs = objectsof(s)
    dim = dimensionof(s)
    ps = _tolist(dim, pvec)
    vs = _tolist(dim, vvec)
    return Space([reconstruct(objs[i], ps[i], vs[i]) for i in 1:length(objs)])
end
