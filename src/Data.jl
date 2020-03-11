module Data

using Random: AbstractRNG, GLOBAL_RNG

function rotate(θ, x)
    # Rotation matrix
    R = [
        cos(θ) -sin(θ); 
        sin(θ)  cos(θ)
    ]
    return R * x
end

add_gaussiannoise(xs, sigma) = add_gaussiannoise(GLOBAL_RNG, xs, sigma)
function add_gaussiannoise(rng::AbstractRNG, xs::AbstractVector{<:AbstractVector}, sigma)
    return map(x -> x + sigma * randn(rng, size(x)...), xs)
end

export rotate, add_gaussiannoise

end # module
