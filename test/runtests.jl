using HullReachNV
using Test
using LazySets

@testset "HullReachNV.jl" begin
    nnet = read_nnet("nnet/test_nnet.nnet")

    solver = HullReach(0.001, false)

    center = [0.5, 0.5]
    radius = [0.5, 0.5]

    in_hyper = Hyperrectangle(center, radius)

    lower = [-3.5, -1.5]
    upper = [-1.5, 2.0]

    out_hyper = Hyperrectangle(low=lower, high=upper)

    problem = Problem(nnet, in_hyper, out_hyper)

    result = solve(solver, problem)

    @test result.status == :violated

end
