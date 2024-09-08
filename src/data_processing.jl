"""
    pig_to_features(file_path::String)::Tuple{Vector{Tuple},Vector{Int}}

Get features (pitch, time_shift, duration) and labels (hands) from PIG fingering file.
"""
function pig_to_features(file_path::String)::Tuple{Vector{Vector{Float32}},Vector{Float32}}
    pre = 0
    features = Vector{Vector{Float32}}()
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

            push!(features, Float32[pitch, time_shift*100 ])#,duration*100])
            push!(labels,Float32(channel))
            pre = onset_time
        end
    end
    return features,labels
end

"""
    midi_to_features(file_path::String)::Vector{Vector{Float32}}

Get features (pitch, time_shift, duration) from MIDI file.
"""
function midi_to_features(midi_file::MIDIFile)::Vector{Vector{Float32}}
    notes = notes_in_first_track(midi_file)

    pre = 0
    result = Vector{Vector{Float32}}()
    for n in notes
        onset_time = metric_time(midi_file,n)
        push!(result,Float32[n.pitch, (onset_time - pre)/10])#, duration_metric_time(midi_file,n)/10])
        pre = onset_time
    end
    return result
end

function notes_in_first_track(midi_file::MIDIFile)
    for t in midi_file.tracks
        notes = getnotes(t)
        if length(notes)!=0
            return notes
        end
    end
end

"""
    get_train_dataloaders(dataset_path::String; batch_size=10, seq_length=20, shuffle=false)::DataLoader

Get train data loader.
"""
function get_train_dataloaders(dataset_path::String; batch_size=10, seq_length=20)::DataLoader
    feature_result = Vector()
    label_result = Vector()
    for file in readdir(dataset_path;join=true)
        features,labels = pig_to_features(file)
        push!(feature_result,stack(stack.(partition(features,seq_length,1))))
        push!(label_result,stack(stack.(partition(labels,seq_length,1))))
    end
    return DataLoader((cat(feature_result...;dims=3),cat(label_result...;dims=2));
        batchsize = batch_size, shuffle=true, parallel = true)
end

"""
    get_val_datas(dataset_path::String)

Get validate datas.
"""
function get_val_datas(dataset_path::String)
    feature_result = Vector()
    label_result = Vector()
    for file in readdir(dataset_path;join=true)
        features,labels = pig_to_features(file)
        features = stack(features)
        push!(feature_result,features)
        push!(label_result,labels)
    end
    return feature_result, label_result
end

"""
    predict_y(y)

Predict left hand or right hand by output.
"""
predict_y(y) = y > 0.5f0 ? 1 : 0

function generate_midi(input_file::String; output_file::String="",
    weight_file=pkgdir(PianoHands,"model","model-0.91502.jld2"),HIDDEN_SIZE=14)
    
    midi_file = load(input_file)
    hand_classify = inferance_midi(midi_file,weight_file,HIDDEN_SIZE)
    
    notes_lh = Notes()
    notes_rh = Notes()
    track_lh = MIDITrack()
    track_rh = MIDITrack()

    new_midi_file = MIDIFile(1,midi_file.tpq,Vector{MIDITrack}())
    for (h,note) in zip(hand_classify,notes_in_first_track(midi_file))
        push!(h == 1 ? notes_lh : notes_rh, note)
    end

    addnotes!(track_lh, notes_lh)
    addtrackname!(track_lh, "piano left")
    addnotes!(track_rh, notes_rh)
    addtrackname!(track_rh, "piano right")
    push!(new_midi_file.tracks, track_lh, track_rh)
    save(isempty(output_file) ? first(splitext(input_file))*"_out.mid" : output_file, new_midi_file)
end