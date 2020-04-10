#HullReach
using HullReachNV
using Test
using LazySets

nnet = read_nnet("nnet/test_nnet.nnet")

solver = HullReach(0.001, false)

center = [0.5, 0.5]
radius = [0.5, 0.5]

in_hyper = Hyperrectangle(center, radius)

lower = [-3.75, -1.5]
upper = [-1.5, 2.0]

out_hyper = Hyperrectangle(low=lower, high=upper)

problem = Problem(nnet, in_hyper, out_hyper)

print("HullReach - test")
timed_result =@timed solve(solver, problem)
print(" - Time: " * string(timed_result[2]) * " s")
print(" - Output: ")
print(timed_result[1].status)
print("\n")
