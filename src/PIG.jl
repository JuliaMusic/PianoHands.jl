function pig_to_pignotes(file_path::String)::Tuple{Vector{Tuple},Vector{Int}}
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