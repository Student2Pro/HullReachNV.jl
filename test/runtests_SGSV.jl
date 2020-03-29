using HullReachNV
using Test
using LazySets

@testset "HullReachNV.jl" begin
    solver = SGSV(0.01)

    acas_nnet = read_nnet("nnet/ACASXU_run2a_4_5_batch_2000.nnet")

    b_lower = [ 0.21466922,  0.11140846, -0.4999999 ,  0.3920202 ,  0.4      ]
    b_upper = [ 0.58819589,  0.4999999 , -0.49840835,  0.66474747,  0.4      ]

    in_hyper  = Hyperrectangle(low = b_lower, high = b_upper)
    inputSet = in_hyper#convert(HPolytope, in_hyper)

    A = [1.0, 0.0, 0.0, 0.0, -1.0]'
    b = [0.0]
    outputSet = HPolytope(A, b)

    problem_acas = Problem(acas_nnet, inputSet, outputSet)

    result = solve(solver, problem_acas)

    @test result.status == :holds

end
