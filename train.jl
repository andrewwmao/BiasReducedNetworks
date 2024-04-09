using Pkg
Pkg.activate(".")
Pkg.instantiate()

using Flux
using Flux.MLUtils
using Flux.Data
using Flux.Losses
using Flux.Optimise
using CUDA #make sure to set to the CUDA version of the GPU
using Base: @kwdef
using LinearAlgebra, Statistics
using MAT, BSON
using ArgParse
using Printf
using ProgressBars
include("NN_functions.jl")

## Helper functions
function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--num_epochs", "-e"
            help = "Number of epochs"
            arg_type = Int
            default = 500
        "--num_coeffs", "-c"
            help = "Number of basis coefficients"
            arg_type = Int
            default = 15
        "--num_params", "-p"
            help = "Number of output parameters"
            arg_type = Int
            default = 6
        "--learning_rate", "-r"
            help = "Learning rate"
            arg_type = Float64
            default = 1e-4
        "--decay_rate", "-g"
            help = "Learning rate decay rate"
            arg_type = Float64
            default = 0.0
        "--batchsize", "-b"
            help = "Batchsize"
            arg_type = Int
            default = 2048
        "--loss_function", "-f"
            help = "Loss function: msecrb or varcon"
            arg_type = String
            default = "msecrb"
        "--num_realizations", "-m"
            help = "Number of noise realizations (measurements) per fingerprint"
            arg_type = Int
            default = 1
        "--lambda", "-l"
            help = "Lambda for composite bias/variance cost"
            arg_type = Float64
            default = 1.0
        "--delta"
            help = "Delta for variance constrained cost"
            arg_type = Float64
            default = 1.0
    end

    return parse_args(s)
end

@kwdef mutable struct Args
    T = Float32                 # floating precision
    η::Float64 = 1e-4           # learning rate
    γ::Float64 = 0.0            # decay rate for learning rate
    λ::Float64 = 1.0            # lambda for bias/variance composite losses
    δ::Float64 = 1.0            # delta for variance constrained loss
    batchsize::Int = 2048       # batch size
    epochs::Int = 500           # number of epochs
    use_cuda::Bool = true       # use gpu (if cuda available)
    Nc::Int = 15                # number of basis coeffs
    Np::Int = 6                 # number of output parameters
    loss::String="msecrb"       # loss function
    normalize::String="coeff1"  # fingerprint normalization strategy, l2 or coeff1
    Ns::Int=Nc                  # number of basis coeffs per segment
    Nr::Int=1                   # number of noise realizations per fingerprint
    pretrained_file::String=""  # pretrained model file name to load, needs to match current model
end

function get_epochTD(sc, loss, batchsize, idx_train, idx_val, device; p, CRB, SNR_min=10, SNR_max=50, normalize="coeff1", Ns=nothing, Nr=1)
    # preprocesses simulated fingerprints and creates dataloaders

    # preprocess fingerprints
    if loss == "msecrb"
        scn, CRBn = preprocess_fingerprints(sc; CRB=CRB, SNR_min=SNR_min, SNR_max=SNR_max, normalize=normalize, Ns=Ns)
        scn = dropdims(scn; dims=3)
    elseif loss == "varcon"
        scn, CRBn = preprocess_fingerprints(sc; CRB=CRB, SNR_min=SNR_min, SNR_max=SNR_max, Nr=Nr, normalize=normalize, Ns=Ns)
        scn = permutedims(scn, (1,3,2))
    end

    # create training & validation dataloaders
    if loss == "msecrb"
        train_loader = DataLoader((scn[:,idx_train], p[:,idx_train], CRBn[:,idx_train]), batchsize=batchsize, shuffle=true)
        val_loader = DataLoader((scn[:,idx_val], p[:,idx_val], CRBn[:,idx_val]), batchsize=batchsize)
    elseif loss == "varcon"
        train_loader = DataLoader((scn[:,:,idx_train], p[:,idx_train], CRBn[:,idx_train]), batchsize=batchsize, shuffle=true)
        val_loader = DataLoader((scn[:,:,idx_val], p[:,idx_val], CRBn[:,idx_val]), batchsize=batchsize)
    end

    return train_loader |> device, val_loader |> device
end

function train(td_file; kws...)
    args = Args(; kws...) # collect options in a struct for convenience

    ## Detect device, use GPU if available
    if CUDA.functional() && args.use_cuda
        @info "Training on GPU"
        CUDA.allowscalar(false)
        device = gpu
    else
        @info "Training on CPU"
        device = cpu
    end
    flush(stderr)

    ## Load the dataset
    @info "Loading the data"
    flush(stderr)
    file = matopen(td_file)
    sc = Complex{args.T}.(read(file, "sc")[1:args.Nc,:])
    p   = args.T.(read(file, "p"))
    CRB = args.T.(read(file, "CRB"))
    close(file)
    Nsamples = size(sc,2)
    Random.seed!(12345)
    idx = randperm(Nsamples)
    Ntrain = Int(round(Int, ceil(Nsamples*0.8) / args.batchsize) * args.batchsize)
    idx_train = idx[1:Ntrain]
    idx_val = idx[Ntrain + 1:end]

    ## Construct model
    @info "Creating model"
    flush(stderr)
    Ncoef = args.Nc*2
    model = build_model(Ncoef) |> device
    println(model)
    flush(stdout)
    if isfile(args.pretrained_file)
        @info "Loading pretrained model file"
        flush(stderr)
        Flux.loadmodel!(model, BSON.load(args.pretrained_file)[:model])
    end
    ps = Flux.params(model) # model's trainable parameters

    ## Training loss
    function msecrb_loss(x::CuArray{T},y::CuArray{T},z::CuArray{T}; agg="mean", Np::Int=args.Np) where {T}
        output = (convert_params(model(x))[1:Np,:] .- y[1:Np,:]).^2 ./ z[1:Np,:]
        if agg == "mean"
            return mean(output)
        elseif agg == "sum"
            return sum(output)
        elseif agg == "sum2"
            return sum(output, dims=2)
        end
    end
    function varcon_loss(x::CuArray{T},y::CuArray{T},z::CuArray{T},λ::Float64,δ::Float64; agg="mean", Np::Int=args.Np, Nr::Int=args.Nr) where {T}
        x = reshape(x, Ncoef, :)
        output = convert_params(model(x))
        output = reshape(output[1:Np,:], Np, Nr, :)
        y = reshape(y[1:Np,:], Np, 1, :)
        z = reshape(z[1:Np,:], Np, 1, :)
        μ = mean(output, dims=2)
        σ = sum((output .- μ).^2 ./ z, dims=2) ./ Nr # uncorrected sample variance weighted by CRB
        if agg == "mean"
            loss = λ * mean(max.(σ .- δ, 0))
            return loss + mean((μ .- y).^2 ./ z) # bias
        elseif agg == "sum2" #return Np x 2 matrix
            loss = λ .* vec(sum(max.(σ .- δ, 0), dims=3))
            loss2 = vec(sum((μ .- y).^2 ./ z, dims=3)) #bias
            return cat(loss, loss2, dims=2)
        end
    end

    ## Optimizer
    opt = RADAM(args.η)
    if args.γ > 0
        opt = Optimiser(opt, InvDecay(args.γ))
    end

    ## Training
    if args.loss == "msecrb"
        epoch_loss = zeros(Float64, args.Np+1, args.epochs)
    elseif args.loss == "varcon"
        epoch_loss = zeros(Float64, args.Np+1, 2, args.epochs)
    end
    iter = ProgressBar(1:args.epochs)
    @info "Starting training"
    @time for epoch in iter
        flush(stderr)

        # get training data with different M0 and noise for every epoch
        train_loader, val_loader = get_epochTD(sc, args.loss, args.batchsize, idx_train, idx_val, device; p=p, CRB=CRB, normalize=args.normalize, Ns=args.Ns, Nr=args.Nr)

        if args.loss == "msecrb"
            for (x,y,z) in train_loader #training
                gs = gradient(() -> msecrb_loss(x, y, z), ps)
                update!(opt, ps, gs)
            end
            for (x,y,z) in val_loader #validation
                epoch_loss[1:args.Np, epoch] .+= msecrb_loss(x, y, z; agg="sum2") |> cpu
            end
        elseif args.loss == "varcon"
            for (x,y,z) in train_loader #training
                gs = gradient(() -> varcon_loss(x, y, z, args.λ, args.δ), ps)
                update!(opt, ps, gs)
            end
            for (x,y,z) in val_loader #validation
                epoch_loss[1:args.Np, :, epoch] .+= varcon_loss(x, y, z, args.λ, args.δ; agg="sum2") |> cpu
            end
        end
        if args.loss == "msecrb"
            epoch_loss[1:args.Np, epoch] ./= length(idx_val) #this handles unequal batches
            epoch_loss[end, epoch] = mean(epoch_loss[1:args.Np, epoch])
        elseif args.loss == "varcon"
            epoch_loss[1:args.Np, :, epoch] ./= length(idx_val) #this handles unequal batches
            epoch_loss[end, :, epoch] = mean(epoch_loss[1:args.Np, :, epoch], dims=1)
        end

        # print current validation loss
        if args.loss == "msecrb"
            postfix_string = @sprintf("%.3f", epoch_loss[end,epoch])
            for i = 1:args.Np
                postfix_string = string(postfix_string, @sprintf(", L[%g]:%.2f", i, epoch_loss[i, epoch]))
            end
        elseif args.loss == "varcon"
            postfix_string = @sprintf("%.3f", sum(epoch_loss[end,:,epoch]))
            for i = 1:args.Np
                postfix_string = string(postfix_string, @sprintf(", L[%g]:%.2f", i, sum(epoch_loss[i,:,epoch])))
            end
        end
        set_postfix(iter, Loss=postfix_string)
    end

    @info "Training complete"
    flush(stderr)
    return cpu(model), epoch_loss
end

## Main file section

# read the command line arguments
parsed_args = parse_commandline()
epochs = parsed_args["num_epochs"]
Nc = parsed_args["num_coeffs"]
Np = parsed_args["num_params"]
η = parsed_args["learning_rate"]
γ = parsed_args["decay_rate"]
batchsize = parsed_args["batchsize"]
loss = parsed_args["loss_function"]
Nr = parsed_args["num_realizations"]
λ = parsed_args["lambda"]
δ = parsed_args["delta"]

# define the training dataset
# td_file = "td.mat"
td_file = "td_test.mat" # for testing the pipeline
if loss == "msecrb"
    pretrained_filename = ""
elseif loss == "varcon"
    # pretrained_filename = "msecrb_Nc$(Nc)_Np$(Np)_epochs500_bs2048_lr1e-04_dr5e-05.bson"
    pretrained_filename = "msecrb_Nc$(Nc)_Np$(Np)_epochs500_bs120_lr1e-04_dr5e-05.bson" # for testing pipeline
end

# start training
@info loss
flush(stderr)
model, epoch_loss = train(td_file; epochs=epochs, Nc=Nc, Np=Np, η=η, γ=γ, batchsize=batchsize, loss=loss,
    Nr=Nr, λ=λ, δ=δ, pretrained_file=pretrained_filename)

# save model and epoch loss
if loss == "varcon"
    # model_filename = "biasreduced_network.bson"
    model_filename = "network_test.bson" # for testing the pipeline
elseif loss == "msecrb"
    model_filename = string(loss, @sprintf("_Nc%g_Np%g_epochs%g_bs%g_lr%.0e_dr%.0e.bson", Nc, Np, epochs, batchsize, η, γ))
end
BSON.@save model_filename model epoch_loss