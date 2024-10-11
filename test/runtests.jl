using Test,PianoHands,MIDI,Lux,Random,Printf,LuxCUDA,Optimisers,Zygote,JLD2

@testset "pig to feature" begin
    # train_piano("../PianoFingeringDataset/dataset/",
    # "../PianoFingeringDataset/testset/";
    # SEQ_LENGTH=65,
    # BATCH_SIZE=12,
    # LEARNING_RATE = 0.0005f0,
    # HIDDEN_SIZE = 14,
    # EVALUATE_PER_N_TRAIN = 50
    # )

    generate_midi("./ymsn_full.mid";weight_file="../model/model-0.92091.jld2")
end