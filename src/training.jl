function build_model(RNNCELL,HIDDEN_SIZE::Int)
    broadcast_layer = @compact(; layer = Dense(2HIDDEN_SIZE=>1,sigmoid)) do x ::Union{NTuple{<:AbstractArray}, AbstractVector{<:AbstractArray}}
        @return map(layer, x)
    end
    return Chain(BidirectionalRNN(RNNCELL(2=>HIDDEN_SIZE)),broadcast_layer)
end

function train_piano(DATASET_PATH,
    TESTSET_PATH;
    BATCH_SIZE = 12,
    SEQ_LENGTH = 75,
    HIDDEN_SIZE = 14,
    LEARNING_RATE = 0.0005f0,
    MAX_EPOCH = 200,
    EVALUATE_PER_N_TRAIN = 50)

    dev = gpu_device()

    # Get the dataloaders
    train_loader = get_train_dataloaders(DATASET_PATH;batch_size=BATCH_SIZE, seq_length=SEQ_LENGTH) |> dev
    val_x, val_y = get_val_datas(TESTSET_PATH)

    # Create the model
    model = build_model(GRUCell,HIDDEN_SIZE)
    display(model)
    rng = Xoshiro(0)
    
    ps, st = Lux.setup(rng, model) |> dev
    train_state = Training.TrainState(model, ps, st,Adam(LEARNING_RATE))

    logitbce = BinaryCrossEntropyLoss();
    loss_fn(ŷ,y) = sum(logitbce.(vec.(ŷ),eachslice(y;dims=1)))
    function compute_loss(model, ps, st, (x, y))
        ŷ, st_ = model(x, ps, st)
        return loss_fn(ŷ,y), st_, (; y_pred=ŷ)
    end

    matches_num(y_pred, y_true) = sum((y_pred .> 0.5f0) .== y_true)

    i = 1
    heightest_acc = 0
    for epoch in 1:MAX_EPOCH
        loss_sum = 0
        # Train the model
        for (x,y) in train_loader
            (_, loss, _, train_state) = Training.single_train_step!(
                AutoZygote(), compute_loss, (x, y), train_state)
            i+=1
            loss_sum += loss
            if i % EVALUATE_PER_N_TRAIN == 0
                @printf "Epoch [%3d]: Loss %4.5f\n" epoch loss_sum/i
                # Validate the model
                st_ = Lux.testmode(train_state.states)
                matchs = 0
                note_count = mapreduce(length,+,val_y)
                loss_sum_in = 0
                for (x, y) in zip(val_x,val_y)
                    x = x |> dev
                    y = y |> dev
                    ŷ, st_ = model(x, train_state.parameters, st_)
                    loss_sum_in += loss_fn(ŷ, y)
                    matchs += matches_num(vcat(ŷ...),y)
                end
                
                acc = round(matchs/note_count;digits=5)
                @printf "Validation: Loss %4.5f Accuracy %4.5f\n" loss_sum_in/15 acc
                
                # save model
                if acc > heightest_acc
                    ps_trained,st_trained = (train_state.parameters, train_state.states) |> cpu_device()
                    rm("trained_model-$(heightest_acc).jld2", force=true)
                    @save "trained_model-$(acc).jld2" {compress = false} ps_trained st_trained
                    heightest_acc = acc
                end
            end
        end
        i = 1
    end
end

function inferance_midi(midi_file::MIDIFile,weight_file::String,HIDDEN_SIZE::Int)::Vector{Int}
    model = build_model(GRUCell,HIDDEN_SIZE)
    display(model)
    dev = gpu_device()
    @load weight_file ps_trained st_trained
    ps_trained,st_trained |> dev

    st_ = Lux.testmode(st_trained)
    y, st_ = model(stack(midi_to_features(midi_file)), ps_trained, st_)
    y |> cpu_device()
    return (predict_y ∘ first).(y)
end