struct BackwardReach end

import LazySets:HPoly

# This is the main function
function solve(solver::BackwardReach, problem::Problem)
    safe_zone = backward_network(solver, problem.network, problem.output)
    if isempty(safe_zone)
        return BasicResult(:unknown)
    end
    print("safe_zone $(dim(safe_zone))\n")
    return issubset(problem.input, safe_zone) ? BasicResult(:holds) : BasicResult(:violated)
end

function backward_network(solver::BackwardReach, nnet::Network, input::HPolyhedron{Float64})
    reach = input
    for i in length(nnet.layers):-1:1
        reach = backward_layer(solver, nnet.layers[i], nnet.layers[i].activation, reach)
    end
    return reach
end

function backward_layer(solver::BackwardReach, layer::Layer, act::Id, input::HPolyhedron{Float64})
    lc = get_constraints(input)
    (W, b) = (layer.weights, layer.bias)
    lco = backward_affine_map(lc, W, b)
    remove_redundant_constraints!(lco)
    return HPolyhedron(lco)
end

function backward_layer(solver::BackwardReach, layer::Layer, act::ReLU, input::HPolyhedron{Float64})
    n_dim = dim(input)
    lc = get_constraints(input)
    (W, b) = (layer.weights, layer.bias)
    lcs = Vector{LinearConstraint{Float64, Vector{Float64}}}(undef, 0)
    diag = zeros(Float64, (n_dim, n_dim))
    b = zeros(Float64, n_dim)
    act_status = zeros(Int64, n_dim)
    n_part = 2 ^ n_dim
    for i in 1:n_part
        n = i
        lci = Vector{LinearConstraint{Float64, Vector{Float64}}}(undef, 0)
        for j in 1:n_dim
            n, act_status[j] = fldmod1(n, 2)
            diag[j, j] = 1.0
            if act_status[j] == 1
                push!(lci, LinearConstraint(diag[j,:], 0.0))
            end
            push!(lci, LinearConstraint(-diag[j,:], 0.0))
        end #setfield! immutable struct of type HalfSpace cannot be changed
        append!(lci, lc)
        if isempty(HPolyhedron(lci))
            continue
        end
        remove_redundant_constraints!(lci) #LinearConstraint
        lci_to_delete = Vector{Int64}(undef, 0)
        for k in 1:length(lci)
            if !in(lci[k], lc)
                push!(lci_to_delete, k)
            end
        end
        deleteat!(lci, lci_to_delete)
        Wi = diag
        bi = b
        for l in 1:n_dim
            Wi[l, l] = act_status[l] == 1 ? 0.0 : 1.0
            bi[l] = 0.0
        end
        append!(lcs, backward_affine_map(lci, Wi, bi))
    end
    if length(lcs) == 0
        ni = size(W, 2)
        diagi = zeros(Float64, (ni, ni))
        for m in 1:ni
            diagi[m, m] = 1.0
            push!(lcs, LinearConstraint(diagi[m,:], 0.0))
        end
        return HPolyhedron(lcs)
    end
    lco = backward_affine_map(lcs, W, b)
    remove_redundant_constraints!(lco)
    return HPolyhedron(lco)
end

function backward_layer(solver::BackwardReach, layer::Layer, act::PiecewiseLinearActivation, input::HPolyhedron{Float64})
    n_dim = dim(input)
    lc = get_constraints(input)
    (W, b) = (layer.weights, layer.bias)
    inflections = get_inflection(act)
    lcs = Vector{LinearConstraint{Float64, Vector{Float64}}}(undef, 0)
    diag = zeros(Float64, (n_dim, n_dim))
    b = zeros(Float64, n_dim)
    if inflections == []
        W1 = diag
        b1 = b
        for l in 1:n_dim
            W1[l, l] = act.pieces[1].slope
            b1[l] = act.pieces[1].intercept
        end
        append!(lcs, backward_affine_map(lc, W1, b1))
    else
        out_bounds = [-Inf; act.(inflections); Inf]
        n_interval = length(inflections) + 1
        act_status = zeros(Int64, n_dim)
        n_part = n_interval ^ n_dim
        for i in 1:n_part
            n = i
            lci = Vector{LinearConstraint{Float64, Vector{Float64}}}(undef, 0)
            for j in 1:n_dim
                n, act_status[j] = fldmod1(n, n_interval)
                diag[j, j] = 1.0
                act_status[j] == n_interval || push!(lci, LinearConstraint(diag[j,:], out_bounds[act_status[j] + 1])) #right bound
                act_status[j] == 1 || push!(lci, LinearConstraint(-diag[j,:], -out_bounds[act_status[j]])) #left bound
            end #setfield! immutable struct of type HalfSpace cannot be changed
            append!(lci, lc)
            if isempty(HPolyhedron(lci))
                continue
            end
            remove_redundant_constraints!(lci) #LinearConstraint
            lci_to_delete = Vector{Int64}(undef, 0)
            for k in 1:length(lci)
                if !in(lci[k], lc)
                    push!(lci_to_delete, k)
                end
            end
            deleteat!(lci, lci_to_delete)
            Wi = diag
            bi = b
            for l in 1:n_dim
                Wi[l, l] = act.pieces[act_status[l]].slope
                bi[l] = act.pieces[act_status[l]].intercept
            end
            append!(lcs, backward_affine_map(lci, Wi, bi))
        end
    end
    lco = backward_affine_map(lcs, W, b)
    remove_redundant_constraints!(lco)
    return HPolyhedron(lco)
end

#ensure LinearConstraint{Float64, Vector{Float64}}
function get_constraints(input::HPolyhedron{Float64})
    C, d = tosimplehrep(input) #Cannot `convert` an object of type HalfSpace{Float64,LazySets.Arrays.SingleEntryVector{Float64}} to an object of type HalfSpace{Float64,Array{Float64,1}}
    lc = Vector{LinearConstraint{Float64, Vector{Float64}}}(undef, length(d))
    for i in 1:length(d)
        lc[i] = LinearConstraint(C[i,:], d[i])
    end
    return lc
end

function backward_affine_map(input::Vector{LinearConstraint{Float64, Vector{Float64}}}, M::Matrix{Float64}, v::Vector{Float64})
    output = Vector{LinearConstraint{Float64, Vector{Float64}}}(undef, 0)
    for lc in input
        vp = vprod(lc.a, M)
        if !iszero(vp)
            push!(output, LinearConstraint(vp, lc.b - lc.a' * v))
        end
    end
    return output
end

function vprod(x::Vector{Float64}, y::Matrix{Float64})
    @assert length(x) == size(y, 1) "Vector length $(length(x)) doesn't match Matrix size $(size(y))"
    n = size(y, 2)
    prod = zeros(Float64, n)
    for i in 1:n
        prod[i] = x' * y[:,i]
    end
    return prod
end
