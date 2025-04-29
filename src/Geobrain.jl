module GeoBrain

# pull in all the pieces
include("manifold.jl")
include("alive.jl")
include("imagination.jl")
include("hypergraph.jl")
include("controller.jl")
include("train.jl")
include("utils.jl")

# re-export the key types & functions
export GeoGraphReal, init_alive_buffers!, oxi_shapes_alive!,
       LivingSimplexTensor, init_living_simplex_tensor, step_imagination!,
       GeoNode, build_hypergraph, MetaController, init_meta_controller,
       train_geobrain!, save_geobrain

end
