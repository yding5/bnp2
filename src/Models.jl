module Models

using Flux, MLToolkit.Neural, MLToolkit.DistributionsX

### Utilites

l1of(x) = sum(abs.(x))
l2of(x) = sum(x.^2)

# L1 and L2 regularization
l1regof(m) = sum(l1of, params(m)) / nparams(m)
l2regof(m) = sum(l2of, params(m)) / nparams(m)

function apply_kernels(X)
    return vcat(X, 1 ./ X, sin.(X), cos.(X))
end

function euclidsq(X::T) where {T<:AbstractMatrix}
    XiXj = transpose(X) * X
    x² = sum(X.^2; dims=1)
    return transpose(x²) .+ x² - 2XiXj
end

function pairwise_compute(X)
    dim = div(size(X, 1), 3)
    X = cat([X[(i-1)*dim+1:i*dim,:] for i in 1:3]...; dims=3)
    hs = map(1:size(X, 2)) do t
        Xt = X[:,t,:]
        Dt = euclidsq(Xt)
        ht = sum(Dt; dims=2)
    end
    return hcat(hs...)
end

### Models

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

function NeuralBodyForce(Hs::NTuple{N,Int}, E::Int, Δt::T, σ::T; n=3, d=2, act=relu) where {N, T<:Float32}
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

function NeuralRelation(Hs::NTuple{N,Int}, E::Int, Δt::T, σ::T; n=3, d=2, act=relu) where {N, T<:Float32}
    D = d * n
    embedding = DenseNet(2d, Hs, E, act)
    relation = DenseNet(2E, Hs, d, act)
    return NeuralRelation(embedding, relation, Δt, [σ])
end

function merge_force(fs, n)
    d, twonB = size(fs)                 # d, 2 * n * B
    B = div(twonB, 2n)
    f = cat(map(1:n) do i               # over n objects
        sum(map(n-1:-1:1) do j          # over n - 1 objects
            idx1 = (2i - j) * B + 1
            idx2 = (2i - j + 1) * B
            f1 = fs[:,idx1:idx2]        # d, B
        end)
    end...; dims=1)                     # d * n, B
    return reshape(f, d, n, :)          # d, n, B
end

"Estimate force by state"
function estforce(nbf::NeuralRelation, s)
    twod, n = size(s)                           # 2d, n, B
    d = div(twod, 2)

    s = reshape(s, twod, :)                     # 2d, n * B
    ebd = nbf.embedding(s)                      # E, n * B
    ebd = reshape(ebd, size(ebd, 1), n, :)      # E, n, B
    
    ebds = cat(                              
        # Body 1
        cat(ebd[:,1,:], ebd[:,2,:]; dims=1),    # 2E, B
        cat(ebd[:,1,:], ebd[:,3,:]; dims=1),    # 2E, B
        # Body 2
        cat(ebd[:,2,:], ebd[:,1,:]; dims=1),    # 2E, B
        cat(ebd[:,2,:], ebd[:,3,:]; dims=1),    # 2E, B
        # Body 3
        cat(ebd[:,3,:], ebd[:,1,:]; dims=1),    # 2E, B
        cat(ebd[:,3,:], ebd[:,2,:]; dims=1),    # 2E, B
        dims=2
    )                                           # 2E, 2B
    fs = nbf.relation(ebds)                     # d, 2 * n * B
    return merge_force(fs, n)                   # d, n, B
end

transit(nbf::NeuralRelation, s) = leapfrog(nbf, s)

function (nbf::NeuralRelation)(s, s′)
    s, s′ = reshape(s, size(s, 1), size(s, 2), :), reshape(s′, size(s′, 1), size(s′, 2), :)
    return lossof(nbf, s, s′)
end

end # module
