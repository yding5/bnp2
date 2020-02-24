# bnpphysics (BNP^2)
BNP for Physics

## A brief description for the proposed model

Task: Given the mass $m$, initial position $\mathbf{q}_0$ and initial velocity $\mathbf{p}_0$ of a particle,
observe the its trajectory $\{\mathbf{q}_t, \mathbf{p}_t\}$ over a time period $t = 1, \dots, T$,
infer the force that is applied to the particle.

The simplest scenario is a 2D world where only a single force $\mathbf{f} \in \mathbb{R}^2$ (plus gravity) is (consistently) applied to the particle during the time period.

The generative process is
$$
\begin{aligned}
m &= 1; \mathbf{q}_0 = [0, 0]; \mathbf{p}_0 = [10, 20] \\
\mathbf{f} &\sim \mathcal{N}(\mathbf{0}, 10\; \mI) \\
\forall t &\in 1, \dots, T \\
& \bar{\mathbf{q}}_t, \bar{\mathbf{p}}_t = \mathrm{transition}(\mathbf{q}_{t-1}, \mathbf{p}_{t-1}, \mathbf{f}, m, 0.1) \\
& \mathbf{q}_t \sim \mathcal{N}(\bar{\mathbf{q}}_t, 0.1\; \mI) \\
& \mathbf{p}_t \sim \mathcal{N}(\bar{\mathbf{p}}_t, 0.1\; \mI)
\end{aligned},
$$
where $\mathrm{transition}(\mathbf{q}, \mathbf{p}, \mathbf{f}, m, \Delta t)$ means integrating the ordinary equations
$$
\begin{aligned}
\frac{d \mathbf{q}}{d t} &= \mathbf{p} \\ 
\frac{d \mathbf{p}}{d t} &= \frac{\mathbf{f} + [0, -9.8]}{m}
\end{aligned}
$$
for a time period of $\Delta t$.

An example of such model implemented in Julia using Turing is

```julia
@model single_force(pos_list, vel_list) = begin
    obj = Particle(1.0, [0.0, 0.0], [10.0, 20.0])
    F ~ MvNormal(zeros(2), 10 * ones(2))
    for i in 1:length(pos_list)
        obj′ = transition(obj, Earth(Force(F)), sim)
        pos_list[i] ~ MvNormal(positionof(obj′), 0.1)
        vel_list[i] ~ MvNormal(velocityof(obj′), 0.1)
        obj = obj′
    end
end
```

The model can be extended by

1. making $\mathbf{f}$ dependent on $t$,
2. making the number of $\mathbf{f}$ potentially infinite (i.e., non-parametric),
3. including more than one objects into $\mathbf{q}$ and $\mathbf{p}$ and
4. introducing more primitive forces.

$$
\begin{aligned}
\mathrm{Given}\;& \mathbf{m}, \mathbf{q}_0, \mathbf{p}_0, \alpha \\
\forall k &\in 1, 2, \dots \\
& f_c^k \sim \mathcal{C}at(K) \\
& \mathbf{a}_p^k \sim \mathcal{N}(\mathbf{0}, \mI) \\
\forall t &\in 1, \dots, T \\
& \mathbf{Z}_t \sim \mathrm{IBP}(\alpha) \\
& \mathbf{f}_t = F(\mathbf{Z}_t, \mathbf{q}_{t-1}, \mathbf{p}_{t-1}, [f_c^1, f_c^2, \dots]) \\
& \bar{\mathbf{q}}_t, \bar{\mathbf{p}}_t = \mathrm{transition}(\mathbf{q}_{t-1}, \mathbf{p}_{t-1}, \mathbf{f}_t, \mathbf{m}, 0.1) \\
& \mathbf{q}_t \sim \mathcal{N}(\bar{\mathbf{q}}_t, 0.1\; \mI) \\
& \mathbf{p}_t \sim \mathcal{N}(\bar{\mathbf{p}}_t, 0.1\; \mI)
\end{aligned},
$$

where $f_c$ is the type of force, e.g. a single force or a magnetic field, $\mathbf{a}_p$ is the corresponding attributes of the force, e.g. the force vector for single force or the strength of the field, and the function $F$ compute the actual force vector that is applied to all objects.
