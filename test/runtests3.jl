using HullReachNV
using Test
using LazySets

@testset "HullReachNV.jl" begin
    solver = HullReach()

    nnet = read_nnet("nnet/small_nnet.nnet")

    in_hyper  = Hyperrectangle(low = [-0.9], high = [0.9])
    out_superset = Hyperrectangle(low = [30.0], high = [80.0])

    problem_holds = Problem(nnet, in_hyper, out_superset)

    result = solve(solver, problem_holds)

    @test result.status == :holds

end
