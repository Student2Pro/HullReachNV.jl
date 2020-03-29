using HullReachNV
using Test
using LazySets

@testset "HullReachNV.jl" begin
    solver = MaxSens(0.1, false)

    acas_nnet = read_nnet("nnet/ACASXU_run2a_4_5_batch_2000.nnet")

    center = [1.0, 1.0, 1.0, 1.0, 1.0]
    radius = [1.0, 1.0, 1.0, 1.0, 1.0]

    in_hyper = Hyperrectangle(center, radius)

    c = [0.0, 0.0, 0.0, 0.0, 0.0]
    r = [10.0, 4.0, 3.5, 4.5, 4.5]

    out_hyper = Hyperrectangle(c, r)

    problem = Problem(acas_nnet, in_hyper, out_hyper)

    result = solve(solver, problem)

    @test result.status == :holds

end
