module Models

using Flux, MLToolkit.Neural, MLToolkit.DistributionsX

### Utilites

l1of(x) = sum(abs.(x))
l2of(x) = sum(x.^2)

# L1 and L2 regularization
l1regof(m) = sum(l1of, params(m)) / nparams(m)
l2regof(m) = sum(l2of, params(m)) / nparams(m)

### Models

"""NeuralForce"""
struct NeuralForce{T1, T2, T3}
    f::T1
    Δt::T2
    σ::T3
end

Flux.@functor NeuralForce

function NeuralForce(Hs::NTuple{N,Int}, Δt::Float32, σ::Float32; n=3, d=2, act=relu) where {N}
    D = n * d
    #     Dense(2D, Hs[1], act)
    #  -> Dense(Hs[1], Hs[2], act)
    #  -> ...
    #  -> Dense(Hs[end-1], Hs[end], act)
    #  -> Dense(Hs[end], D)
    f = DenseNet(2D, Hs, D, act)
    return NeuralForce(f, Δt, [σ])
end                                 

(nf::NeuralForce)(state) = nf.f(state)

# TODO: Make this variational to account uncertainty in integration
function leapfrog(nf::NeuralForce, state)
    D = div(size(state, 1), 2)
    q, p = state[1:D,:], state[D+1:end,:]
    p = p + nf.Δt / 2 * nf(vcat(q, p))
    q = q + nf.Δt * p
    p = p + nf.Δt / 2 * nf(vcat(q, p))
    return vcat(q, p)
end

function lossof(nf::NeuralForce, s, s′)
    ŝ = leapfrog(nf, s)
    lps = logpdf(Normal(ŝ, nf.σ), s′)    # elementwise log probabilities
    lp = mean(sum(lps; dims=1)) # sum over state (dim 1) and avgerage over time (dim 2)
    l2 = l2regof(nf)
    return (loss=-lp + l2, lp=lp, l2=l2)
end

"""VariationalNeuralForce"""
struct VariationalNeuralForce{T1, T2, T3}
    f::T1
    Δt::T2
    σ::T3
end

Flux.@functor VariationalNeuralForce

function VariationalNeuralForce(Hs::NTuple{N,Int}, Δt::Float32, σ::Float32; n=3, d=2, act=relu) where {N}
    D = n * d
    #     Dense(2D, Hs[1], act)
    #  -> Dense(Hs[1], Hs[2], act)
    #  -> ...
    #  -> Dense(Hs[end-1], Hs[end], act)
    #  -> Dense(Hs[end], 2D)
    f = DenseNet(2D, Hs, 2D, act)
    return VariationalNeuralForce(f, Δt, [σ])
end 

(vnf::VariationalNeuralForce)(state) = vnf.f(state)

end # module
