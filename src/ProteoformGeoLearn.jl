module ProteoformGeoLearn

#=
Core module for learning proteoform geometry on a Boolean-lattice state space.
Exports:
  - ProteoformGraph: custom graph type carrying structure + sparse representation
  - update_geometry!: compute c‑Ricci from a probability vector
  - featured_graph: wrap geometry into a FeaturedGraph for GNN
  - ProteoformGNN: GCN model constructor
  - train!: training loop integrated with simulation
=#

include("geometry.jl")
include("simulation.jl")

export ProteoformGraph, update_geometry!, featured_graph, ProteoformGNN, train!

using Graphs, SparseArrays
using Flux, Flux.Losses: mse
using GeometricFlux, GraphSignals
using Distances, Distributions

# 1) Boolean-lattice graph type
struct ProteoformGraph
    g::SimpleGraph{Int}
    sg::GraphSignals.SparseGraph{Float32}
    pf_states::Vector{String}
    flat_pos::Dict{String,Tuple{Float64,Float64}}
    edges::Vector{Tuple{String,String}}
end

function ProteoformGraph(pf_states::Vector{String},
                         flat_pos::Dict{String,Tuple{Float64,Float64}},
                         edges::Vector{Tuple{String,String}})
    n = length(pf_states)
    g = SimpleGraph(n)
    idx = Dict(s=>i for (i,s) in enumerate(pf_states))
    for (u,v) in edges
        add_edge!(g, idx[u], idx[v])
    end
    A = Float32.(adjacency_matrix(g))
    sg = GraphSignals.SparseGraph(sparse(A), true, Float32)
    return ProteoformGraph(g, sg, pf_states, flat_pos, edges)
end

# 2) Geometry wrapper\ nfunction update_geometry!(pg::ProteoformGraph, ρ::Vector{Float64})
    _, _, C_R_vals, _ = update_geometry_from_rho(ρ, pg.pf_states, pg.flat_pos, pg.edges)
    return C_R_vals
end

# 3) FeaturedGraph constructor
function featured_graph(pg::ProteoformGraph, C_R::Vector{Float32})
    nf = reshape(C_R, 1, :)
    ef = zeros(Float32, 0, ne(pg.g))
    pf = zeros(Float32, 0, ne(pg.g))
    return FeaturedGraph(pg.sg, nf, ef, nothing, pf, :adjm)
end

# 4) GNN model
function ProteoformGNN(hidden::Int=16)
    Chain(
        GCNConv(1 => hidden, relu),
        GCNConv(hidden => 1),
        x -> reshape(x, :)
    )
end

function (m::Chain)(pg::ProteoformGraph, ρ::Vector{Float32})
    C_Rf = Float32.(update_geometry!(pg, Float64.(ρ)))
    fg = featured_graph(pg, C_Rf)
    ρ̂ = m(fg)
    ρ̂ ./= sum(ρ̂)
    return ρ̂
end

# 5) Training loop
function train!(model, pg::ProteoformGraph, ρ0::Vector{Float32}, T::Int;
                opt=ADAM(), verbose=true)
    ps = Flux.params(model)
    ρ = copy(ρ0)
    for t in 1:T
        ρ_gt = copy(ρ)
        oxi_shapes_alive!(ρ_gt, pg.pf_states, pg.flat_pos, pg.edges; max_moves=10)
        loss() = mse(model(pg, ρ), ρ_gt)
        grads = gradient(loss, ps)
        Flux.Optimise.update!(opt, ps, grads)
        oxi_shapes_alive!(ρ, pg.pf_states, pg.flat_pos, pg.edges; max_moves=10)
        verbose && @info "step=$(t) loss=$(loss())"
    end
end

end # module ProteoformGeoLearn
