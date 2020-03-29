struct BackwardReach end

import LazySets:HPoly

# This is the main function
function solve(solver::BackwardReach, problem::Problem)
    safe_zone = backward_network(solver, problem.network, problem.output)
    if issubset(problem.input, safe_zone)
        return BasicResult(:holds)
    end
    return BasicResult(:violated)
end

function backward_network(solver::BackwardReach, nnet::Network, input::HPolyhedron{Float64})
    reach = input
    for i in length(nnet.layers):-1:1
        reach = backward_layer(solver, nnet.layers[i], reach)
    end
    return reach
end

function backward_layer(solver::BackwardReach, layer::Layer, input::HPolyhedron{Float64})
    n_dim = dim(input)
    C, d = tosimplehrep(input) #Cannot `convert` an object of type HalfSpace{Float64,LazySets.Arrays.SingleEntryVector{Float64}} to an object of type HalfSpace{Float64,Array{Float64,1}}
    lc = Vector{LinearConstraint{Float64, Vector{Float64}}}(undef, length(d))
    for h in 1:length(d)
        lc[h] = LinearConstraint(C[h,:], d[h])
    end #ensure LinearConstraint{Float64, Vector{Float64}}
    (W, b, act) = (layer.weights, layer.bias, layer.activation)
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
        for lc1 in lc
            v = vprod(lc1.a, W1)
            if !iszero(v) #a half-space needs a non-zero normal vector
                push!(lcs, LinearConstraint(v, lc1.b - lc1.a' * b1))
            end
        end
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
            for lcc in lci
                v = vprod(lcc.a, Wi)
                if !iszero(v)
                    push!(lcs, LinearConstraint(v, lcc.b - lcc.a' * bi))
                end
            end
        end
    end
    lco = Vector{LinearConstraint{Float64, Vector{Float64}}}(undef, 0)
    for lcj in lcs
        v = vprod(lcj.a, W)
        if !iszero(v)
            push!(lco, LinearConstraint(v, lcj.b - lcj.a' * b))
        end
    end
    remove_redundant_constraints!(lco)
    return HPolyhedron(lco)
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
