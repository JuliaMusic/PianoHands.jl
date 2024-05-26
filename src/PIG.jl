"""
    pig_to_features(file_path::String)::Tuple{Vector{Tuple},Vector{Int}}

Get features (pitch, time_shift, duration) and labels (hands) from PIG fingering file.
"""
function pig_to_features(file_path::String)::Tuple{Vector{Tuple},Vector{Int}}
    pre = 0
    features = Vector{Tuple}()
    labels = Vector{Int}()
    open(file_path) do file
        for ln in eachline(file)
            if startswith(ln,"//") || length(ln) == 0
                continue
            end
            _, onset_time, offset_time, spelled_pitch, _, _, channel, _ = split(ln,"\t")
            onset_time = parse(Float32,onset_time)
            offset_time = parse(Float32,offset_time)
            channel = parse(Int,channel)

            time_shift = onset_time - pre
            duration = offset_time - onset_time
            pitch = name_to_pitch(spelled_pitch)

            push!(features,(pitch, time_shift, duration))
            push!(labels,channel)
            pre = onset_time
        end
    end
    return features,labels
end

function get_data(features::Vector{Tuple},labels::Vector{Int};seq_length = 20, batch_size = 10)
    data_features = Flux.chunk(collect.(partition(collect.(features),seq_length,1)); size = batch_size)
    data_features = [permutedims(d,(2,1,3)) for d in stack.([stack.(d) for d in data_features])]
    
    labels = [[l] for l in labels]
    data_labels = Flux.chunk(collect.(partition(labels,seq_length,1)); size = batch_size)
    data_labels = [permutedims(d,(2,1,3)) for d in stack.([stack.(d) for d in data_labels])]
    return data_features,data_labels
end