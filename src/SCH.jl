"""
    SCH(tolerance::Float64)

SCH,SGSV Combine HullReach, performs over-approximated reachability analysis to compute the over-approximated output reachable set for a network.

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
"""
@with_kw struct SCH
    tolerance::Float64 = 1.0
end

# This is the main function
function solve(solver::SCH, problem::Problem)
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
                    if isborder(sections[i], problem.input)
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

#to determine whether x has intersection with any border of y
function isborder(x::Hyperrectangle, y::HPolytope)
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
        l = length(dy)
        point1 = zeros(Float64, n)
        point2 = zeros(Float64, n)
        for j in 1:l
            for k in 1:n
                if Cy[j, k] >= 0.0
                    point1[k] = upper[k]
                    point2[k] = lower[k]
                else
                    point1[k] = lower[k]
                    point2[k] = upper[k]
                end
            end
            result1 = sum(point1 .* Cy[j,:]) - dy[j]
            result2 = sum(point2 .* Cy[j,:]) - dy[j]
            if result1 >= 0.0 && result2 <= 0.0
                return true
            end
        end
        return false
    end
end

function forward_layer(solver::SCH, L::Layer, input::Hyperrectangle)
    (W, b, act) = (L.weights, L.bias, L.activation)
    center = zeros(size(W, 1))
    gamma  = zeros(size(W, 1))
    for j in 1:size(W, 1)
        node = Node(W[j,:], b[j], act)
        center[j], gamma[j] = forward_node(solver, node, input)
    end
    return Hyperrectangle(center, gamma)
end

function forward_node(solver::SCH, node::Node, input::Hyperrectangle)
    output    = node.w' * input.center + node.b
    deviation = sum(abs.(node.w) .* input.radius)
    βmax = node.act(output + deviation)
    βmin = node.act(output - deviation)
    return ((βmax + βmin)/2, (βmax - βmin)/2)
end
