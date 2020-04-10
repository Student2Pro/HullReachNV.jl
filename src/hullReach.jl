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
    result = true
    delta = solver.resolution
    lower, upper = low(problem.input), high(problem.input)
    hull_lower, hull_upper = deepcopy(lower), deepcopy(upper)
    for i in 1:length(lower)
        hull_lower[i] = upper[i]
        hull_upper[i] = lower[i]
        lower_hull = Hyperrectangle(low = lower, high = hull_upper)
        upper_hull = Hyperrectangle(low = hull_lower, high = upper)
        if !check_reach(solver, lower_hull, problem.network, problem.output)
            result = false
        end
        if !check_reach(solver, upper_hull, problem.network, problem.output)
            result = false
        end
        hull_lower[i] = lower[i]
        hull_upper[i] = upper[i]
    end
    if result
        return BasicResult(:holds)
    end
    return BasicResult(:violated)
end

function check_reach(solver::HullReach, input::Hyperrectangle, nnet::Network, output::Hyperrectangle)
    result = true
    delta = solver.resolution
    lower, upper = low(input), high(input)
    n_hypers_per_dim = max.(ceil.(Int, (upper-lower) / delta), 1)

    # preallocate work arrays
    local_lower, local_upper, CI = similar(lower), similar(lower), similar(lower)
    for i in 1:prod(n_hypers_per_dim)
        n = i
        for j in firstindex(CI):lastindex(CI)
            n, CI[j] = fldmod1(n, n_hypers_per_dim[j])
        end
        @. local_lower = lower + delta * (CI - 1)
        @. local_upper = min(local_lower + delta, upper)
        hyper = Hyperrectangle(low = local_lower, high = local_upper)
        reach = forward_network(solver, nnet, hyper)
        if !issubset(reach, output)
            result = false
        end
    end
    return result
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
