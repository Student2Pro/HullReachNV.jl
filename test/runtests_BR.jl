using HullReachNV
using Test
using LazySets

@testset "HullReachNV.jl" begin
    solver = BackwardReach()

    acas_nnet = read_nnet("nnet/ACASXU_run2a_4_5_batch_2000.nnet")

    center = [1.0, 1.0, 1.0, 1.0, 1.0]
    radius = [1.0, 1.0, 1.0, 1.0, 1.0]

    in_hyper = Hyperrectangle(center, radius)

    c = [0.0, 0.0, 0.0, 0.0, 0.0]
    r = [20.0, 20.0, 20.0, 20.0, 20.0]

    out_hyper = Hyperrectangle(c, r)

    problem = Problem(acas_nnet, convert(HPolyhedron, in_hyper), convert(HPolyhedron, out_hyper))

    result = solve(solver, problem)

    @test result.status == :holds

end
