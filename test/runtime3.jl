#SGSV, Specification-Guided Safety Verification
using HullReachNV
using Test
using LazySets

#acas_nnet = read_nnet("nnet/ACASXU_run2a_4_5_batch_2000.nnet")

#solver = SGSV(0.000061036) #0.000061036
nnet = read_nnet("nnet/test_nnet.nnet")

solver = SGSV(0.001)

center = [0.5, 0.5]
radius = [0.5, 0.5]

in_hyper = Hyperrectangle(center, radius)

lower = [-3.75, -1.5]
upper = [-1.5, 2.0]

out_hyper = Hyperrectangle(low=lower, high=upper)

problem = Problem(nnet, in_hyper, out_hyper)

print("SGSV - test")
timed_result =@timed solve(solver, problem)
print(" - Time: " * string(timed_result[2]) * " s")
print(" - Output: ")
print(timed_result[1].status)
print("\n")
