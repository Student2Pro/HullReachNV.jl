"""
    SGSV(tolerance::Float64)

SGSV, Specification-Guided Safety Verification, performs over-approximated reachability analysis to compute the over-approximated output reachable set for a network.

# Problem requirement
1. Network: any depth, any activation that is monotone
2. Input: `HPolytope`
3. Output: `HPolytope`

# Return
`BasicResult`

# Method
Search and Reachability

# Property
Sound but not complete.

# Reference
[W. Xiang, H.-D. Tran, and T. T. Johnson,
"Specification-Guided Safety Verification for Feedforward Neural Networks,"
*ArXiv Preprint ArXiv:1812.06161*, 2018.](https://arxiv.org/abs/1812.06161)
"""
@with_kw struct SGSV
    tolerance::Float64 = 1.0
end

# This is the main function
function solve(solver::SGSV, problem::Problem)
    input = overapproximate(problem.input)
    stack = Vector{Hyperrectangle}(undef, 0)
    push!(stack, input)
    while !isempty(stack)
        interval = pop!(stack)
        reach = forward_network(solver, problem.network, interval)
        if issubset(reach, problem.output)
            continue
        else
            if get_largest_width(interval) > solver.tolerance
                sections = bisect(interval)
                for i in 1:2
                    if isintersection(sections[i], problem.input)
                        push!(stack, sections[i])
                    end
                end
            else
                return BasicResult(:unknown)
            end
        end
    end
    return BasicResult(:holds)
end

function get_largest_width(input::Hyperrectangle)
    width = high(input) - low(input)
    return max(width...)
end

#interval refinement
function bisect(input::Hyperrectangle)
    lower, upper = low(input), high(input)
    width = upper - lower
    index = 0
    temp = 0
    for i in 1:length(width)
        if width[i] >= temp
            temp = width[i]
            index = i
        end
    end
    upper[index] = input.center[index]
    lower_part = Hyperrectangle(low = lower, high = upper)
    lower[index] = input.center[index]
    upper[index] = input.center[index] + input.radius[index]
    upper_part = Hyperrectangle(low = lower, high = upper)
    return (lower_part, upper_part)
end

function isintersection(x::Hyperrectangle, y::HPolytope)
    lower, upper = low(x), high(x)
    n = length(lower)
    Cy, dy = tosimplehrep(y)
    Cx = zeros(Float64, 2n, n)
    dx = zeros(Float64, 2n)
    for i in 1:n
        Cx[2i, i] = 1.0
        dx[2i] = upper[i]
        Cx[2i-1, i] = -1.0
        dx[2i-1] = -lower[i]
    end
    C = vcat(Cy, Cx)
    d = vcat(dy, dx)
    if isempty(HPolytope(C, d))
        return false
    else
        return true
    end
end

function forward_layer(solver::SGSV, L::Layer, input::Hyperrectangle)
    (W, b, act) = (L.weights, L.bias, L.activation)
    center = zeros(size(W, 1))
    gamma  = zeros(size(W, 1))
    for j in 1:size(W, 1)
        node = Node(W[j,:], b[j], act)
        center[j], gamma[j] = forward_node(solver, node, input)
    end
    return Hyperrectangle(center, gamma)
end

function forward_node(solver::SGSV, node::Node, input::Hyperrectangle)
    output    = node.w' * input.center + node.b
    deviation = sum(abs.(node.w) .* input.radius)
    βmax = node.act(output + deviation)
    βmin = node.act(output - deviation)
    return ((βmax + βmin)/2, (βmax - βmin)/2)
end
