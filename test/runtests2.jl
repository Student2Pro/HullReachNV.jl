using LazySets

center = [1.0, 1.0, 1.0, 1.0, 1.0]
radius = [1.0, 1.0, 1.0, 1.0, 1.0]
input = Hyperrectangle(center, radius)
delta = 1.0

n_dim = dim(input)
lower, upper = low(input), high(input)
radius = fill(delta/2, n_dim)

hyperrectangle_list = ceil.(Int, (upper - lower)/delta)

part_list = zeros(Int64, n_dim)
part_list[1] = 1
for j in 2:n_dim
    part_list[j] = part_list[j-1] * max(hyperrectangle_list[j-1], 1)
end

n_hyperrectangle = part_list[n_dim] * max(hyperrectangle_list[n_dim], 1)

hyperrectangles = Vector{Hyperrectangle}(undef, n_hyperrectangle)
for k in 1:n_hyperrectangle
    n = k - 1
    center = Vector{Float64}(undef, n_dim)
    for i in n_dim:-1:1
        id = div(n, part_list[i])
        n = mod(n, part_list[i])
        lower_cell = lower[i] + min(delta, input.radius[i]*2) * id
        radius[i] = min(delta, upper[i] - lower_cell) * 0.5
        center[i] = lower_cell + radius[i];
    end
    hyperrectangles[k] = Hyperrectangle(center[:], radius[:])
end
return hyperrectangles
