# PianoHands.jl
Predicting hand assignments in piano MIDI using neural networks

# Use pre-trained model

``` julia
using PianoHands
generate_midi("./your_midi.mid";)
```

You will get a midi file `your_midi_out.mid`, track 1 is left hand notes, track 2 is right hand notes.

# Train Your own model.

## Dataset preparation

Download PIG v1.2 Dataset to `PianoFingeringDataset` and remove duplicate fingering file, approximately 150 fingering files are required.

``` julia
function train_piano(DATASET_PATH,
    TESTSET_PATH;
    BATCH_SIZE = 12,
    SEQ_LENGTH = 75,
    HIDDEN_SIZE = 14,
    LEARNING_RATE = 0.0005f0,
    MAX_EPOCH = 200,
    EVALUATE_PER_N_TRAIN = 50)
```

The network structure is bi-directional GRU + Dense, and the hidden layer size can be adjusted by parameters. There is no stopping condition for training, you need stop manually.

Use trained weight:

```julia
generate_midi(input_file::String;
    output_file="",
    weight_file=pkgdir(PianoHands,"model","model-0.91502.jld2"),
    HIDDEN_SIZE=14)
```

If you need to change the input featears, or the network structure, you must download this package for local debugging.