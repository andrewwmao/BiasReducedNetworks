using Random
using LinearAlgebra
using Flux
using Flux.NNlib

# custom activation functions and NN layers
struct Split{T}
    paths::T
end
Split(paths...) = Split(paths)
Flux.@functor Split
(m::Split)(x::AbstractArray) = tuple(map(f->f(x), m.paths))

BlockBN(Nin, Nout) = Chain(Dense(Nin, Nout), BatchNorm(Nout, relu)) #Dense + batchnorm
SkipBN(In, N) = Chain(SkipConnection(In, +), BatchNorm(N, relu)) #skip connection w/ batchnorm

relu(x) = max(0, x)
relu6(x) = min(max(0, x), 6)

fhsig(x) = max(hardsigmoid(x), oftype(x, 1e-5)) #for m0s
fhsig2(x) = max(2 * hardsigmoid(x), oftype(x, 1e-2))
fhsig5(x) = max(5 * hardsigmoid(x), oftype(x, 1e-2))
fhsig10(x) = max(10 * hardsigmoid(x), oftype(x, 1e-2))
fhsig25(x) = max(25 * hardsigmoid(x), oftype(x, 1e-2))
fhsig50(x) = max(50 * hardsigmoid(x), oftype(x, 1e-2))
fhsigT2s_wide(x) = max(oftype(x, 30e-6) * hardsigmoid(x), oftype(x, 1e-6))
hsw0(x) = hardsigmoid(x) #for w0
hsB1(x) = oftype(x, 0.6) * hardsigmoid(x) + oftype(x, 0.6) # for B1

# Bounds on parameter outputs
BoundedOut(N) = Split(Dense(N,1,fhsig), Dense(N,1,fhsig2), Dense(N,1,fhsig25), Dense(N,1,fhsig50), Dense(N,1,fhsig10), Dense(N,1,fhsigT2s_wide), Dense(N,1,hsw0), Dense(N,1,hsB1)) #m0s, R1f, R2f, Rx, R1s, T2s, w0, B1

#helper functions
function build_model(Nc::Int=30)
    return Chain(
        SkipBN(Chain(BlockBN(Nc,256),
        SkipBN(Chain(BlockBN(256,512),
        SkipBN(Chain(BlockBN(512,1024), BlockBN(1024,768), Dense(768,512)), 512),
        BlockBN(512,384), Dense(384,256)), 256),
        BlockBN(256,128), Dense(128,Nc)), Nc),
        BlockBN(Nc,32),
        BoundedOut(32))
end

function convert_params(model_output)
    return reduce(vcat, collect(model_output[1])) #convert tuple of ntuple to a matrix of size [params, batchsize]
end

function preprocess_fingerprints(sc::Matrix{Complex{T}}; CRB=nothing, SNR_min=10, SNR_max=100, Nr::Int=1, rand_phase::Bool=true, rand_SNR::Bool=false, normalize::String="coeff1", Ns::Int=0) where {T}
    # adds random phase and noise before compressing to basis and concatenating real/imag

    if !isnothing(CRB)
        @assert size(sc)[2:end] == size(CRB)[end-length(size(sc))+2:end] "Check that size of fingerprints and CRBs are the same"
    end
    if length(size(sc)) > 2
        sc = reshape(sc, (size(sc,1), :))
        if !isnothing(CRB)
            CRB = reshape(CRB, (size(CRB, 1), :))
        end
    elseif length(size(sc)) == 1
        sc = reshape(sc, length(sc), 1)
        if !isnothing(CRB) && length(size(CRB)) == 1
            CRB = reshape(CRB, length(CRB), 1)
        end
    end
    Nsamples = size(sc,2)
    Nc = size(sc,1)

    #iterate over noise realizations
    scn = zeros(T, Nc*2, Nsamples, Nr)
    rng = MersenneTwister(12345);
    M0_phase = 1
    if SNR_min != Inf
        if !rand_SNR # pick a random SNR per fingerprint that's the same across Nr
            M0_mag = (SNR_max - SNR_min) .* rand(rng, T, 1, Nsamples) .+ SNR_min #easier to uniformly sample SNR range with PD than sigma
        end
        if !rand_phase && normalize != "coeff1" #one random phase for all noise realizations
            M0_phase = exp.(1im * 2 * π * rand(rng, T, 1, Nsamples))
        end
    end
    scn_temp = [zeros(Complex{T}, size(sc)) for _ = 1:Threads.nthreads()]
    if !isnothing(CRB)
        if rand_SNR
            CRBn = zeros(T, size(CRB)..., Nr)
        else
            if length(size(CRB)) == 2
                CRBn = CRB ./ M0_mag.^2
            elseif length(size(CRB)) == 3
                CRBn = CRB ./ reshape(M0_mag, 1, 1, Nsamples).^2
            end
        end
    end
    Threads.@threads for i = 1:Nr
        idt = Threads.threadid() # TODO: fix data race bug
        if SNR_min != Inf
            if rand_phase && normalize != "coeff1" #new random phase per noise realization
                M0_phase = exp.(1im * 2π * rand(rng, T, 1, Nsamples))
            end
            if rand_SNR
                M0_mag = (SNR_max - SNR_min) .* rand(rng, T, 1, Nsamples) .+ SNR_min
            end
            # add noise for SNR_max ∈ [10,100]
            scn_temp[idt] .= sc .* M0_phase .* M0_mag
            scn_temp[idt] .+= randn(rng, T, Nc, Nsamples) .+ 1im .* randn(rng, T, Nc, Nsamples)
        end

        #normalize fingerprints based on l2 norm or first coefficient
        normalize_fingerprints!(scn_temp[idt]; normalize, Ns)

        #concatenate real and imaginary parts
        @views scn[1:Nc,:,i]     .= real.(scn_temp[idt])
        @views scn[Nc+1:end,:,i] .= imag.(scn_temp[idt])

        if !isnothing(CRB) && rand_SNR
            if length(size(CRB)) == 2
                @views CRBn[:,:,i] .= CRB ./ M0_mag.^2
            elseif length(size(CRB)) == 3
                @views CRBn[:,:,:,i] .= CRB ./ M0_mag.^2
            end
        end
    end
    GC.gc()

    if !isnothing(CRB)
        return scn, CRBn
    else
        return scn
    end
end

function normalize_fingerprints!(sc::AbstractArray; normalize::String="coeff1", Ns::Int=0)
    shape = size(sc)
    if Ns == 0
        Ns = shape[1]
    end
    if rem(size(sc,1), Ns) == 0 && Int(size(sc,1) / Ns) > 1 #normalize per Ns segment
        num_segs = shape[1] ÷ Ns
        sc = reshape(sc, Ns, num_segs, :)
        for j ∈ axes(sc, 2)
            if normalize == "l2"
                @views sc[:,j,:] ./= reshape(norm.(eachcol(sc[:,j,:]), 2), 1, shape[2])
            elseif normalize == "coeff1"
                @views sc[:,j,:] ./= reshape(sc[1,j,:], 1, shape[2])
            end
        end
        sc = reshape(sc, shape[1], :)
    else
        if normalize == "l2"
            sc ./= reshape(norm.(eachcol(sc), 2), 1, shape[2])
        elseif normalize == "coeff1"
            sc ./= reshape(@view(sc[1,:]), 1, shape[2])
        end
    end
end

function fit_invivo(model, x::AbstractArray{T}, mask; normalize::String="coeff1", Ns::Int=0, sbatch=10_000) where T
    shape = size(x)
    Ncoef = shape[end]
    img_shape = shape[1:end-1]

    m0s = Array{real(T)}(undef, img_shape)
    R1f = similar(m0s)
    R2f = similar(m0s)
    Rx  = similar(m0s)
    R1s = similar(m0s)
    T2s = similar(m0s)
    m0s .= 0
    R1f .= 0
    R2f .= 0
    Rx  .= 0
    R1s .= 0
    T2s .= 0

    sc = transpose(x[mask,:])
    normalize_fingerprints!(sc; normalize, Ns)

    scat = Array{real(T)}(undef, 2Ncoef, sum(mask))
    scat[    1:end÷2,:] .= real.(sc)
    scat[end÷2+1:end,:] .= imag.(sc)

    nbatch = ceil(Int, sum(mask) / sbatch)
    Threads.@threads for ibatch = 0 : nbatch-1
        idx = sbatch * ibatch + 1 : min(sbatch * (ibatch + 1), sum(mask))
        qM = model(@view scat[:,idx])[1]
        @views m0s[mask][idx] = qM[1]
        @views R1f[mask][idx] = qM[2]
        @views R2f[mask][idx] = qM[3]
        @views  Rx[mask][idx] = qM[4]
        @views R1s[mask][idx] = qM[5]
        @views T2s[mask][idx] = qM[6]
    end

    return m0s, R1f, R2f, Rx, R1s, T2s
end