module Models

include("Models/utilites.jl")

using Flux, MLToolkit.Neural, MLToolkit.DistributionsX

abstract type AbstractTransitionModel end

function lossof(m::AbstractTransitionModel, s, s′)
    ŝ = transit(m, s)
    lps = logpdf(Normal(ŝ, m.σ), s′)    # elementwise log probabilities
    loglikelihood = mean(
        sum(lps; dims=1:ndims(lps)-1)   # sum over state (dim 1:end-1) and 
    )                                   # avgerage over rest (dim end)
    return (loss=-loglikelihood, loglikelihood=loglikelihood)
end

"""DeepTransition"""
struct DeepTransition{T1, T2} <: AbstractTransitionModel
    f::T1
    σ::T2
end

Flux.@functor DeepTransition

function DeepTransition(Hs::NTuple{N,Int}, σ::T; n=3, d=2, act=relu) where {N, T<:Float32}
    D = n * d
    #     Dense(2D, Hs[1], act)
    #  -> Dense(Hs[1], Hs[2], act)
    #  -> ...
    #  -> Dense(Hs[end-1], Hs[end], act)
    #  -> Dense(Hs[end], 2D)
    f = DenseNet(2D, Hs, 2D, act)
    return DeepTransition(f, [σ])
end

transit(dt::DeepTransition, s) = dt.f(s[1:end,:])

function (dt::DeepTransition)(s, s′)
    s, s′ = reshape(s, prod(size(s)[1:2]), :), reshape(s′, prod(size(s′)[1:2]), :)
    return lossof(dt, s, s′)
end

abstract type AbstractForceModel <: AbstractTransitionModel end

function leapfrog(nf::AbstractForceModel, state)
    d = div(size(state, 1), 2)
    q, p = state[1:d,Base.tail(axes(state))...], state[d+1:end,Base.tail(axes(state))...]
    p = p + nf.Δt / 2 * estforce(nf, cat(q, p; dims=1))
    q = q + nf.Δt * p
    p = p + nf.Δt / 2 * estforce(nf, cat(q, p; dims=1))
    return cat(q, p; dims=1)
end

"""NeuralForce"""
struct NeuralForce{T1, T2, T3} <: AbstractForceModel
    f::T1
    Δt::T2
    σ::T3
end

Flux.@functor NeuralForce

function NeuralForce(Hs::NTuple{N,Int}, Δt::T, σ::T; n=3, d=2, act=relu) where {N, T<:Float32}
    D = n * d
    f = DenseNet(2D, Hs, D, act)
    return NeuralForce(f, Δt, [σ])
end

"Estimate force by state"
function estforce(nf::NeuralForce, s)
    twod, n = size(s)
    d = div(twod, 2)
    s = cat([s[:,i,:] for i in 1:size(s, 2)]...; dims=1)
    f = nf.f(s)
    return reshape(f, d, n, :)
end

transit(nf::NeuralForce, s) = leapfrog(nf, s)

function (nf::NeuralForce)(s, s′)
    s, s′ = reshape(s, size(s, 1), size(s, 2), :), reshape(s′, size(s′, 1), size(s′, 2), :)
    return lossof(nf, s, s′)
end

"""NeuralBodyForce"""
struct NeuralBodyForce{T1, T2, T3, T4} <: AbstractForceModel
    embedding::T1
    f::T2
    Δt::T3
    σ::T4
end

Flux.@functor NeuralBodyForce

function NeuralBodyForce(E::Int, Hs::NTuple{N,Int}, Δt::T, σ::T; n=3, d=2, act=relu) where {N, T<:Float32}
    D = d * n
    embedding = DenseNet(2d, Hs, E, act)
    f = DenseNet(E * n, Hs, d * n, act)
    return NeuralBodyForce(embedding, f, Δt, [σ])
end

"Estimate force by state"
function estforce(nbf::NeuralBodyForce, s)
    twod, n = size(s)                       # 2d, n, B
    d = div(twod, 2)
    s = reshape(s, twod, :)                 # 2d, n * B
    ebd = nbf.embedding(s)                  # E, n * B
    ebd = reshape(ebd, size(ebd, 1) * n, :) # E * n, B
    f = nbf.f(ebd)
    return reshape(f, d, n, :)              # d, n, B
end

transit(nbf::NeuralBodyForce, s) = leapfrog(nbf, s)

function (nbf::NeuralBodyForce)(s, s′)
    s, s′ = reshape(s, size(s, 1), size(s, 2), :), reshape(s′, size(s′, 1), size(s′, 2), :)
    return lossof(nbf, s, s′)
end

"""NeuralRelation"""
struct NeuralRelation{T1, T2, T3, T4} <: AbstractForceModel
    embedding::T1
    relation::T2
    Δt::T3
    σ::T4
end

Flux.@functor NeuralRelation

function NeuralRelation(E::Int, Hs::NTuple{N,Int}, Δt::T, σ::T; n=3, d=2, act=relu) where {N, T<:Float32}
    embedding = Dense(2d, E)
    relation = DenseNet(2E, Hs, d, act)
    return NeuralRelation(embedding, relation, Δt, [σ])
end

"""
    f = merge_force(pwf, n)

Merge pairwise force `pwf` of shape (d, n * (n - 1) * B) into `f` of shape (d, n, B)
by summing over the (n - 1) forces applied to each object.
"""
function merge_force(pwf, n)
    d, twonB = size(pwf)                # d, n * (n - 1) * B
    B = div(twonB, 2n)
    f = cat(map(1:n) do i               # over n objects
        sum(map(n-1:-1:1) do j          # over n - 1 objects
            idx1 = (2i - j) * B + 1
            idx2 = (2i - j + 1) * B
            fij = pwf[:,idx1:idx2]      # d, B
        end)
    end...; dims=1)                     # d * n, B
    return reshape(f, d, n, :)          # d, n, B
end

"""
    rangeexclude(n, i)

Returns range `1:n` with `i` excluded.

```julia
rangeexclude(4, 2)  # => (1, 3, 4)
```
"""
rangeexclude(n, i) = tuple((1:i-1)..., (i+1:n)...)

"""
    pwebd_i = cat_pairwise(ebd, i)

Concat pairwise embeddings for element `i` in `ebd`,
where `ebd` has a shape of (E, n, B) and 
the 2nd dimension is assumed to be the element dimension.
The returned `pwebd` has a shape of (2E, (n - 1) * B).
"""
function cat_pairwise(x, i)
    n = size(x, 2)
    return cat(map(
        j -> cat(x[:,i,:], x[:,j,:]; dims=1),   # 2E, B
        rangeexclude(n, i)                      # over (n - 1) other objects
    )...; dims=2)
end

"Estimate force by state"
function estforce(nbf::NeuralRelation, s)
    twod, n = size(s)                       # 2d, n, B
    d = div(twod, 2)
    s = reshape(s, twod, :)                 # 2d, n * B
    ebd = nbf.embedding(s)                  #  E, n * B
    ebd = reshape(ebd, size(ebd, 1), n, :)  #  E, n, B
    # Pairwise embedding
    pwebd = cat(
        map(
            i -> cat_pairwise(ebd, i),      # 2E, (n - 1) * B
            1:n                             # over n objects
        )...; 
        dims=2
    )                                       # 2E, n * (n - 1) * B
    # Pairwise force by pairwise embedding
    pwf = nbf.relation(pwebd)               #  d, n * (n - 1) * B
    return merge_force(pwf, n)              #  d, n, B
end

transit(nbf::NeuralRelation, s) = leapfrog(nbf, s)

function (nbf::NeuralRelation)(s, s′)
    s, s′ = reshape(s, size(s, 1), size(s, 2), :), reshape(s′, size(s′, 1), size(s′, 2), :)
    return lossof(nbf, s, s′)
end

end # module
