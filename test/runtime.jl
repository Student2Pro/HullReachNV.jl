using HullReachNV, LazySets, Test, LinearAlgebra
import HullReachNV: ReLU, Id

macro no_error(ex)
    quote
        try $(esc(ex))
            true
        catch
            false
        end
    end
end

mnist1 = read_nnet("nnet/mnist1.nnet")
mnist2 = read_nnet("nnet/mnist2.nnet")
mnist3 = read_nnet("nnet/mnist3.nnet")
mnist4 = read_nnet("nnet/mnist4.nnet")

input_center = [0.0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,121,254,136,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,13,230,253,248,99,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,118,253,253,225,42,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,61,253,253,253,74,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,32,206,253,253,186,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,211,253,253,239,69,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,254,253,253,133,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,142,255,253,186,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,149,229,254,207,21,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,54,229,253,254,105,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,152,254,254,213,26,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,112,251,253,253,26,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,29,212,253,250,149,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,36,214,253,253,137,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,75,253,253,253,59,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,93,253,253,189,17,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,253,253,84,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,43,235,253,126,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,99,248,253,119,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,225,235,49,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
output_center = [-1311.1257826380004,4633.767704436501,-654.0718535670002,-1325.349417307,1175.2361184373997,-1897.8607293569007,-470.3405972940001,830.8337987382,-377.7467076115001,572.3674015264198]

in_epsilon = 1.0 #0-255
out_epsilon = 10.0 #logit domain

input_low = input_center .- in_epsilon
input_high = input_center .+ in_epsilon

output_low = output_center .- out_epsilon
output_high = output_center .+ out_epsilon

inputSet = Hyperrectangle(low=input_low, high=input_high)
outputSet = Hyperrectangle(low=output_low, high=output_high)


problem_mnist = Problem(mnist1, inputSet, outputSet)

solver = MaxSens(resolution = 0.01)
print("MaxSens - mnist1")
timed_result =@timed solve(solver, problem_mnist)
print(" - Time: " * string(timed_result[2]) * " s")
print(" - Output: ")
print(timed_result[1])
print("\n")

solver = HullReach(resolution = 0.01)
print("HullReach - mnist1")
timed_result =@timed solve(solver, problem_mnist)
print(" - Time: " * string(timed_result[2]) * " s")
print(" - Output: ")
print(timed_result[1])
print("\n")

"""
problem_mnist = Problem(mnist2, inputSet, outputSet)

solver = MaxSens(resolution = 0.1)
print("MaxSens - mnist2")
timed_result =@timed solve(solver, problem_mnist)
print(" - Time: " * string(timed_result[2]) * " s")
print(" - Output: ")
print(timed_result[1])
print("\n")

solver = HullReach(resolution = 0.1)
print("HullReach - mnist2")
timed_result =@timed solve(solver, problem_mnist)
print(" - Time: " * string(timed_result[2]) * " s")
print(" - Output: ")
print(timed_result[1])
print("\n")


problem_mnist = Problem(mnist3, inputSet, outputSet)

solver = MaxSens(resolution = 0.1)
print("MaxSens - mnist3")
timed_result =@timed solve(solver, problem_mnist)
print(" - Time: " * string(timed_result[2]) * " s")
print(" - Output: ")
print(timed_result[1])
print("\n")

solver = HullReach(resolution = 0.1)
print("HullReach - mnist3")
timed_result =@timed solve(solver, problem_mnist)
print(" - Time: " * string(timed_result[2]) * " s")
print(" - Output: ")
print(timed_result[1])
print("\n")


problem_mnist = Problem(mnist4, inputSet, outputSet)

solver = MaxSens(resolution = 0.1)
print("MaxSens - mnist4")
timed_result =@timed solve(solver, problem_mnist)
print(" - Time: " * string(timed_result[2]) * " s")
print(" - Output: ")
print(timed_result[1])
print("\n")

solver = HullReach(resolution = 0.1)
print("HullReach - mnist4")
timed_result =@timed solve(solver, problem_mnist)
print(" - Time: " * string(timed_result[2]) * " s")
print(" - Output: ")
print(timed_result[1])
print("\n")
"""
