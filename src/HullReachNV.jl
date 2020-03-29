module HullReachNV

using LazySets, LazySets.Approximations
using Polyhedra, CDDLib

using LinearAlgebra
using Parameters
using Interpolations # only for PiecewiseLinear

import LazySets: dim, HalfSpace # necessary to avoid conflict with Polyhedra

using Requires

include("activation.jl")
include("network.jl")
include("problem.jl")
include("util.jl")

function __init__()
  @require Flux="587475ba-b771-5e3f-ad9e-33799f191a9c" include("flux.jl")
end

export
    Solver,
    Network,
    AbstractActivation,
    #PolytopeComplement,
    #complement,
    # NOTE: not sure if exporting these is a good idea as far as namespace conflicts go:
    # ReLU,
    # Max,
    # Id,
    GeneralAct,
    PiecewiseLinear,
    LinearPieces,
    PiecewiseLinearActivation,
    Problem,
    Result,
    BasicResult,
    CounterExampleResult,
    AdversarialResult,
    ReachabilityResult,
    read_nnet,
    solve,
    forward_network,
    check_inclusion

export solve

include("reachability.jl")
include("maxSens.jl")
include("hullReach.jl")
include("SGSV.jl")
include("SCH.jl")
include("pointSearch.jl")
include("backwardReach.jl")
export MaxSens, HullReach, SGSV, SCH, PointSearch, BackwardReach

end # module
