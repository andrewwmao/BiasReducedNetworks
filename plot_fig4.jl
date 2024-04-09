using Pkg
Pkg.activate(".")
Pkg.instantiate()

include("NN_functions.jl")
using MAT, BSON
using Plots
gr()

## load networks and coefficient images
model_biased   = BSON.load("data/msecrb_network.bson")[:model]
model_unbiased = BSON.load("data/biasreduced_network.bson")[:model]
x    = matread("data/coeffs.mat")["x"]
mask = matread("data/coeffs.mat")["mask"]

## run NNs
m0s_biased  , R1f_biased  , R2f_biased,   Rex_biased  , R1s_biased  , T2s_biased   = fit_invivo(model_biased,   x, mask)
m0s_unbiased, R1f_unbiased, R2f_unbiased, Rex_unbiased, R1s_unbiased, T2s_unbiased = fit_invivo(model_unbiased, x, mask)

## plot qMT fitting results
plot(
    heatmap(  m0s_biased, c=:gist_earth, clim=(0,0.30),     aspect_ratio=1, colorbar=false),
    heatmap(  R1f_biased, c=:gist_earth, clim=(0.1,0.7),    aspect_ratio=1, colorbar=false),
    heatmap(  R2f_biased, c=:gist_earth, clim=(8,18),       aspect_ratio=1, colorbar=false),
    heatmap(  Rex_biased, c=:gist_earth, clim=(8,25),       aspect_ratio=1, colorbar=false),
    heatmap(  R1s_biased, c=:gist_earth, clim=(1,5),        aspect_ratio=1, colorbar=false),
    heatmap(  T2s_biased, c=:gist_earth, clim=(8e-6,20e-6), aspect_ratio=1, colorbar=false),
    heatmap(m0s_unbiased, c=:gist_earth, clim=(0,0.30),     aspect_ratio=1, colorbar=false),
    heatmap(R1f_unbiased, c=:gist_earth, clim=(0.1,0.7),    aspect_ratio=1, colorbar=false),
    heatmap(R2f_unbiased, c=:gist_earth, clim=(8,18),       aspect_ratio=1, colorbar=false),
    heatmap(Rex_unbiased, c=:gist_earth, clim=(8,25),       aspect_ratio=1, colorbar=false),
    heatmap(R1s_unbiased, c=:gist_earth, clim=(1,5),        aspect_ratio=1, colorbar=false),
    heatmap(T2s_unbiased, c=:gist_earth, clim=(8e-6,20e-6), aspect_ratio=1, colorbar=false),
    layout=(2,6), size=(1400,500), axis=([], false), dpi=300)