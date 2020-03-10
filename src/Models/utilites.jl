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
