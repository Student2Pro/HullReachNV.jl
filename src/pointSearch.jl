struct LinearPieces
    left::Float64
    right::Float64
    slope::Float64
    intercept::Float64
end

#PiecewiseLinearActivation
struct PiecewiseLinearActivation <: ActivationFunction
    pieces::Vector{LinearPieces}
end

function (f::PiecewiseLinearActivation)(x)
    for p in f.pieces
        if p.left <= x < p.right
            return p.slope * x + p.intercept
        end
    end
end

l1 = LinearPieces(-Inf, 0.0, 0.0, 0.0)
l2 = LinearPieces(0.0, Inf, 1.0, 0.0)
ReLUact = PiecewiseLinearActivation([l1, l2])
IDact = PiecewiseLinearActivation([LinearPieces(-Inf, Inf, 1.0, 0.0)])

struct PointSearch end

# This is the main function
function solve(solver::PointSearch, problem::Problem)
    layers = problem.network.layers
    input = tovrep(problem.input)
    z1_list = Vector{Vector{Float64}}(undef, 0)
    (W, b, act) = (layers[1].weights, layers[1].bias, layers[1].activation)
    for z0 in points(input)
        z1_hat = affine_map(W, b, z0)
        z1 = act.(z1_hat)
        if is_interior_point(forward_network(solver, problem.network, z1, 1), problem.output)
            push!(z1_list, z1)
        else
            return BasicResult(:violated)
        end
    end
    for i in 1:length(layers)
        (W, b, act) = (layers[i].weights, layers[i].bias, layers[i].activation)
        v_list = vertices_list(input) #z_i-1
        zi_hat_list = affine_map.(v_list)
        zi_hat_tope = VPolytope(zi_hat_list)
        zi_list = Vector{Vector{Float64}}(undef, 0)
        inflections = get_inflection(act) #Vector
        if !isempty(inflections)
            C, d = tosimplehrep(zi_hat_tope)
            n = length(d)
            Ci = [C; zeros(Float64, (2, size(C, 2)))]
            di = [d; 0.0; 0.0]
            for j in 1:length(b)
                if j > 1
                    Ci[n+1,j-1] = 0.0
                    Ci[n+2,j-1] = 0.0
                end
                Ci[n+1,j] = 1.0
                Ci[n+2,j] = -1.0
                for k in length(inflections)
                    di[n+1] = inflections[k]
                    di[n+2] = -inflections[k]
                    intersect = HPolytope(Ci, di)
                    if !isempty(intersect)
                        v_inter = tovrep(intersect)
                        for zi_hat in points(v_inter)
                            zi = act.(zi_hat)
                            if is_interior_point(forward_network(solver, problem.network, zi, i), problem.output)
                                push!(zi_list, zi)
                            else
                                return BasicResult(:violated)
                            end
                        end
                    end
                end
            end
        end
        if i < length(layers)
            if i == 1
                append!(zi_list, z1_list)
                z1_list = 0 #free memory
            else
                append!(zi_list, act.(zi_hat_list))
            end
            input = remove_redundant_vertices(VPolytope(zi_list))
        end
    end
    return BasicResult(:holds)
end

function get_inflection(act::PiecewiseLinearActivation)
    n = length(act.pieces)
    inflections = Vector{Float64}(undef, n-1)
    for i in 1:n-1
        inflections[i] = act.pieces[i].right
    end
    return inflections
end

function get_inflection(act::ReLU)
    return [0.0]
end

function get_inflection(act::Id)
    return []
end

function forward_network(solver::PointSearch, nnet::Network, input::Vector{Float64}, last::Int)
    if last == length(nnet.layers)
        return input
    end
    reach = input
    start = last + 1
    for i in start:length(nnet.layers)
        reach = forward_layer(solver, nnet.layers[i], reach)
    end
    return reach
end

function forward_layer(solver::PointSearch, layer::Layer, input::Vector{Float64})
    return layer.activation.(affine_map(layer.weights, layer.bias, input))
end

function affine_map(W::Matrix{Float64}, b::Vector{Float64}, input::Vector{Float64})
    n = length(b)
    output = Vector{Float64}(undef, n)
    for j in 1:n
        output[j] = sum(W[j,:] .* input) + b[j]
    end
    return output
end
