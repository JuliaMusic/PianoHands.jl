using Test,PianoHands

@testset "pig_to_pignotes" begin
    feature, label = PianoHands.pig_to_pignotes("../PianoFingeringDataset/dataset/001-1_fingering.txt")
    @show feature[2]
    @show label[2]
end