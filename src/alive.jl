module GeoBrain.Alive

using ..Manifold: GeoGraphReal, update_real_geometry!
using Random
using Statistics: mean

export init_alive_buffers!, oxi_shapes_alive!

"""
    init_alive_buffers!(G::GeoGraphReal, bitcounts::Vector{Int})

Pre-allocate the integer buffers needed by oxi_shapes_alive!:  
  • counts::Vector{Int}  
  • inflow_int::Vector{Int}  
  • outflow_int::Vector{Int}  
  • deg_pen::Vector{Float32}
"""
function init_alive_buffers!(G::GeoGraphReal, bitcounts::Vector{Int})
    n = G.n
    counts     = Vector{Int}(undef, n)
    inflow_int = zeros(Int, n)
    outflow_int= zeros(Int, n)
    R_total    = length(bitcounts)
    binom      = Dict(k => binomial(R_total, k) for k in 0:R_total)
    deg_pen    = Float32[1f0/binom[bitcounts[i]] for i in 1:n]
    return (
      counts      = counts,
      inflow_int  = inflow_int,
      outflow_int = outflow_int,
      deg_pen     = deg_pen
    )
end

"""
    oxi_shapes_alive!(ρ::Vector{Float32}, G::GeoGraphReal, buffers;
                     max_moves::Int=10)

Perform up to `max_moves` single-molecule transfers on ρ (treated as 100 total),
using a Metropolis-style rule based on curvature & degeneracy.
Updates ρ in place.
"""
function oxi_shapes_alive!(
    ρ::Vector{Float32},
    G::GeoGraphReal,
    buffers;
    max_moves::Int = 10
)
    n           = G.n
    counts      = buffers.counts
    inflow_int  = buffers.inflow_int
    outflow_int = buffers.outflow_int
    deg_pen     = buffers.deg_pen

    # 1) convert ρ → integer counts summing to 100
    @inbounds for i in 1:n
        counts[i] = round(Int, ρ[i]*100f0)
    end
    counts[n] = 100 - sum(counts[1:end-1])

    # 2) recompute geometry on current ρ
    update_real_geometry!(G, ρ)

    # 3) zero the flow buffers
    fill!(inflow_int,  0)
    fill!(outflow_int, 0)

    total_moves = rand(0:max_moves)
    nonzero     = findall(>(0), counts)

    for _ in 1:total_moves
        isempty(nonzero) && break
        i    = rand(nonzero)
        nbrs = G.neighbors[i]
        isempty(nbrs) && continue

        # compute weights to each neighbor
        wsum = 0f0
        ws   = Float32[]
        push!(ws, 0f0)  # dummy to align indices
        for j in nbrs
            ΔS = 0.1f0 + 0.01f0 + abs(G.R_vals[j]) + deg_pen[j]
            Δf = exp(counts[i]/100f0) - G.R_vals[i] + ΔS
            w  = exp(-Δf) * exp(-G.anisotropy[j])
            wsum += w
            push!(ws, w)
        end

        wsum < 1e-8f0 && continue

        # sample a neighbor
        r, cum = rand()*wsum, 0f0
        chosen = first(nbrs)
        for (k,j) in enumerate(nbrs)
            cum += ws[k+1]
            if cum ≥ r
                chosen = j
                break
            end
        end

        # record transfer i → chosen
        inflow_int[chosen]  += 1
        outflow_int[i]     += 1

        # update nonzero list
        if counts[i] - outflow_int[i] == 0
            deleteat!(nonzero, findfirst(==(i), nonzero))
        end
        if counts[chosen] + inflow_int[chosen] == 1
            push!(nonzero, chosen)
        end
    end

    # 4) apply flows and write back to ρ
    @inbounds for i in 1:n
        counts[i] += inflow_int[i] - outflow_int[i]
        ρ[i]       = counts[i]/100f0
    end

    return ρ
end

end # module
