function orthonormalvecof(v::AbstractVector)
    @argcheck length(v) == 2
    return [v[2], -v[1]] / sqrt(sum(v.^2))
end

### Phyics

using PhysicalConstants: CODATA2014, CODATA2018

const G = CODATA2018.NewtonianConstantOfGravitation.val
const g = CODATA2014.StandardAccelerationOfGravitation.val

function attractive_acceleration(m, r²)
    @argcheck !iszero(r²) "Attractive acceleration is undefined for 0 distance."
    return G * m / r²
end
attractive_force(m1, m2, r²) = m1 * attractive_acceleration(m2, r²)
