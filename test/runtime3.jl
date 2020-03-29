#SGSV, Specification-Guided Safety Verification
using HullReachNV
using Test
using LazySets

acas_nnet = read_nnet("nnet/ACASXU_run2a_4_5_batch_2000.nnet")

solver = SGSV(0.01)

center = [1.0, 1.0, 1.0, 1.0, 1.0]
radius = [1.0, 1.0, 1.0, 1.0, 1.0]

in_hyper = Hyperrectangle(center, radius)

lower = [0.0, 0.0, 0.0, 0.0, 0.0]
upper = [10.0, 10.0, 10.0, 10.0, 10.0]

out_hyper = Hyperrectangle(low=lower, high=upper)

problem = Problem(acas_nnet, convert(HPolytope, in_hyper), convert(HPolytope, out_hyper))

print("SGSV - ACAS")
timed_result =@timed solve(solver, problem)
print(" - Time: " * string(timed_result[2]) * " s")
print(" - Output: ")
print(timed_result[1].status)
print("\n")
