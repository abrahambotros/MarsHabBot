#!/bin/env julia

# install the POMDPs.jl interface
Pkg.clone("https://github.com/sisl/POMDPs.jl.git")

# install the SARSOP wrapper
Pkg.clone("https://github.com/sisl/SARSOP.jl")
# build SARSOP, it builds from source, so this may take some time
Pkg.build("SARSOP")

# install the QMDP solver
Pkg.clone("https://github.com/sisl/QMDP.jl")

# install two helper modules
Pkg.clone("https://github.com/sisl/POMDPToolbox.jl") # this provides implementations of discrete belief updating
Pkg.clone("https://github.com/sisl/POMDPDistributions.jl") # helps with sampling

# install discrete value iteration
Pkg.clone("https://github.com/sisl/DiscreteValueIteration.jl")

# install MCTS
Pkg.clone("https://github.com/sisl/MCTS.jl")

# update
Pkg.update()
