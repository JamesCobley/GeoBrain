# examples/generate_dataset.jl
# Example script to generate a proteoform dataset using ProteoformGeoLearn

using ProteoformGeoLearn
using BSON  # for saving intermediate results

# 1) Define the Boolean-lattice state space
const pf_states = ["000","001","010","100","011","101","110","111"]
const flat_pos = Dict(
    "000" => (0.0,3.0),  "001" => (-2.0,2.0),  "010" => (0.0,2.0),
    "100" => (2.0,2.0),  "011" => (-1.0,1.0),  "101" => (0.0,1.0),
    "110" => (1.0,1.0),  "111" => (0.0,0.0)
)
const edges = [
    ("000","001"),("000","010"),("000","100"),
    ("001","011"),("001","101"),("010","011"),
    ("010","110"),("100","110"),("100","101"),
    ("011","111"),("101","111"),("110","111")
]

# 2) Construct the ProteoformGraph once
pg = ProteoformGraph(pf_states, flat_pos, edges)

# 3) Generate systematic initials (e.g. 50 total) and call the ODE evolution
initials = generate_systematic_initials(num_total=50, min_val=0.01)

# 4) Parameters
t_span = 1:100      # time steps
total_samples = 500 # number of trajectories
save_every = 100    # checkpoint interval

# 5) Run evolution and capture metadata
println("Generating dataset...")
X, Y, Metadata, Geos = create_dataset_ODE_alive(
    t_span = t_span,
    max_samples = total_samples,
    save_every = save_every
)

# 6) Save final dataset
BSON.@save "proteoform_dataset.bson" X Y Metadata Geos
println("Dataset generation complete: proteoform_dataset.bson saved.")
