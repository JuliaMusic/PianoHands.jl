module PianoHands

using MIDI, IterTools, Lux, MLUtils, Random, Printf, LuxCUDA, Optimisers, Zygote, JLD2

include("./training.jl")
include("./data_processing.jl")

export train_piano, generate_midi

greet() = print("Hello World!")


end # module PianoHands
