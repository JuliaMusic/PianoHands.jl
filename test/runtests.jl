using Test,PianoHands

@testset "pig to feature" begin
    feature, label = PianoHands.pig_to_features("../PianoFingeringDataset/dataset/001-1_fingering.txt")
    feature, label = PianoHands.get_data(feature,label)
    @show size(feature[1])
    @show size(label[1])
    
end