using DrWatson
@quickactivate "BNP2"
using Test

@testset "Tests" begin
    tests = [
        "world",
        "simulators",
    ]

    foreach(tests) do t
        @eval module $(Symbol("Test", t))
            include($(projectdir("test", t)) * ".jl")
        end
    end
end
