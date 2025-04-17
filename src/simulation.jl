# src/simulation.jl

using StatsBase: sample, Weights
using Random
using LinearAlgebra

# --- Entropy Cost Function (for transitions) ---
function compute_entropy_cost(i::Int, j::Int, C_R_vals::Vector{Float64}, pf_states::Vector{String})
    baseline_DeltaE = 1.0
    mass_heat = 0.1
    reaction_heat = 0.01 * baseline_DeltaE
    conformational_cost = abs(C_R_vals[j])
    # count number of '1's in the binary state string
    deg = Dict(0=>1,1=>3,2=>3,3=>1)[count(c->c=='1', pf_states[j])]
    degeneracy_penalty = 1.0 / deg
    return mass_heat + reaction_heat + conformational_cost + degeneracy_penalty
end

# --- Oxidation‐Shapes Alive Simulation Step ---
function oxi_shapes_alive!(ρ::Vector{Float64}, pf_states::Vector{String}, flat_pos::Dict{String,Tuple{Float64,Float64}}, edges::Vector{Tuple{String,String}}; max_moves::Int=10)
    idx = Dict(s=>i for (i,s) in enumerate(pf_states))
    neighbor_indices = Dict(s=>Int[] for s in pf_states)
    for (u,v) in edges
        push!(neighbor_indices[u], idx[v])
        push!(neighbor_indices[v], idx[u])
    end

    # convert probabilities to counts (sum to 100)
    counts = round.(ρ .* 100.0)
    counts[end] = 100 - sum(counts[1:end-1])
    ρ .= counts ./ 100.0

    # compute geometry updates
    points3D, R_vals, C_R_vals, anis_vals = update_geometry_from_rho(ρ, pf_states, flat_pos, edges)

    inflow = zeros(Float64, length(pf_states))
    outflow = zeros(Float64, length(pf_states))

    total_moves = rand(0:max_moves)
    candidates = findall(x->x>0, counts)

    for _ in 1:total_moves
        isempty(candidates) && break
        i = rand(candidates)
        nbrs = neighbor_indices[pf_states[i]]
        if isempty(nbrs)
            inflow[i] += 0.01
            continue
        end
        # compute move probabilities
        ps = [exp(-((exp(ρ[i]) - C_R_vals[i] + compute_entropy_cost(i,j,C_R_vals,pf_states))) - anis_vals[j]) for j in nbrs]
        normp = sum(ps)
        if normp < 1e-8
            inflow[i] += 0.01
            continue
        end
        ps ./= normp
        chosen = sample(nbrs, Weights(ps))
        inflow[chosen] += 0.01
        outflow[i] += 0.01
    end

    # reconcile inflow/outflow and renormalize
    for i in eachindex(ρ)
        inflow[i] += (counts[i]/100.0) - outflow[i]
    end
    inflow .= max.(inflow, 0.0)
    inflow ./= sum(inflow)
    ρ .= inflow
end
