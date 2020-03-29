"""
    HullReach(resolution::Float64, tight::Bool)

HullReach performs over-approximated reachability analysis to compute the over-approximated output reachable set for a network.

# Problem requirement
1. Network: any depth, any activation that is monotone
2. Input: `Hyperrectangle` or `HPolytope`
3. Output: `AbstractPolytope`

# Return
`ReachabilityResult`

# Property
Sound but not complete.
"""
@with_kw struct HullReach
    resolution::Float64 = 1.0
    tight::Bool         = false
end

# This is the main function
function solve(solver::HullReach, problem::Problem)
    inputs = part(problem.input, solver.resolution)
    f_n(x) = forward_network(solver, problem.network, x)
    outputs = map(f_n, inputs)
    return check_inclusion(outputs, problem.output)
end

function part(input::Hyperrectangle, delta::Float64)
    n_dim = dim(input)
    lower, upper = low(input), high(input)
    radius = fill(delta/2, n_dim)
    n_each_dim = ceil.(Int, (upper - lower)/delta)

    part_list = zeros(Int64, n_dim)
    part_list[1] = 1
    for j in 2:n_dim
        part_list[j] = part_list[j-1] * max(n_each_dim[j-1], 1)
    end

    n_hyperrectangle = part_list[n_dim] * max(n_each_dim[n_dim], 1)
    n_hr = n_hyperrectangle - prod(max.(n_each_dim .- 2, 0))
    hyperrectangles = Vector{Hyperrectangle}(undef, n_hr)
    next = 1
    center = Vector{Float64}(undef, n_dim)
    id = Vector{Int64}(undef, n_dim)
    for k in 1:n_hyperrectangle
        n = k - 1
        border = false

        for i in n_dim:-1:1
            id[i] = div(n, part_list[i])
            if id[i] == 0 || n_each_dim[i] - id[i] == 1
                border = true
            end
            n = mod(n, part_list[i])
        end

        if border == true
            for i in n_dim:-1:1
                lower_cell = lower[i] + min(delta, input.radius[i]*2) * id[i]
                radius[i] = min(delta, upper[i] - lower_cell) * 0.5
                center[i] = lower_cell + radius[i];
            end
            hyperrectangles[next] = Hyperrectangle(center[:], radius[:])
            next += 1
        else
            continue
        end
    end
    return hyperrectangles
end

function part(input::HPolytope, delta::Float64)
    @info "HullReach overapproximates HPolytope input sets as Hyperrectangles."
    part(overapproximate(input), delta)
end

# This function is called by forward_network
function forward_layer(solver::HullReach, L::Layer, input::Hyperrectangle)
    (W, b, act) = (L.weights, L.bias, L.activation)
    center = zeros(size(W, 1))
    gamma  = zeros(size(W, 1))
    for j in 1:size(W, 1)
        node = Node(W[j,:], b[j], act)
        center[j], gamma[j] = forward_node(solver, node, input)
    end
    return Hyperrectangle(center, gamma)
end

function forward_node(solver::HullReach, node::Node, input::Hyperrectangle)
    output    = node.w' * input.center + node.b
    deviation = sum(abs.(node.w) .* input.radius)
    β    = node.act(output)  # TODO expert suggestion for variable name. beta? β? O? x?
    βmax = node.act(output + deviation)
    βmin = node.act(output - deviation)
    if solver.tight
        return ((βmax + βmin)/2, (βmax - βmin)/2)
    else
        return (β, max(abs(βmax - β), abs(βmin - β)))
    end
end
