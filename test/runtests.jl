using Test, ProteoformGeoLearn

# ℹ️ Define the same pf_states, flat_pos and edges you use in src
const pf_states = ["000","001","010","100","011","101","110","111"]
const flat_pos = Dict(
  "000"=>(0.0,3.0), "001"=>(-2.0,2.0), "010"=>(0.0,2.0),
  "100"=>(2.0,2.0), "011"=>(-1.0,1.0), "101"=>(0.0,1.0),
  "110"=>(1.0,1.0), "111"=>(0.0,0.0)
)
const edges = [
  ("000","001"),("000","010"),("000","100"),
  ("001","011"),("001","101"),("010","011"),
  ("010","110"),("100","110"),("100","101"),
  ("011","111"),("101","111"),("110","111")
]

@testset "Basic ProteoformGeoLearn smoke tests" begin
    # construct the graph
    pg = ProteoformGeoLearn.ProteoformGraph(pf_states, flat_pos, edges)

    # single forward pass
    ρ0 = fill(1f0/8f0, 8)
    ρ1 = ProteoformGeoLearn.ProteoformGNN()(pg, ρ0)
    @test length(ρ1) == 8
    @test isapprox(sum(ρ1), 1f0; atol=1e-6)

    # one training step (just check no errors)
    ProteoformGeoLearn.train!(ProteoformGeoLearn.ProteoformGNN(), pg, ρ0, 1)
end
