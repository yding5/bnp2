const G = CODATA2018.NewtonianConstantOfGravitation.val
const g = CODATA2014.StandardAccelerationOfGravitation.val

attractive_force(m1, m2, r²) = G * m1 * m2 / r²
gravityof(m) = m * g
