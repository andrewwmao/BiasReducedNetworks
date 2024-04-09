using Pkg
Pkg.activate(".")
Pkg.instantiate()

include("NN_functions.jl")
using MRIgeneralizedBloch
using MAT, BSON
using Statistics, HypothesisTests
using LaTeXStrings, Measures
using Plots, StatsPlots
gr(display_type=:inline)
theme(:default)
cur_colors = theme_palette(:auto);

## load flip angle pattern
TR = 3.5e-3
control = matread("control.mat")
α = [reshape(control["alpha"],:,6)[:,i] for i = 1:size(reshape(control["alpha"],:,6),2)]
TRF = [reshape(control["TRF"],:,6)[:,i] for i = 1:size(reshape(control["TRF"],:,6),2)]
R2slT = precompute_R2sl()

## load basis and NNs
Ncoef = 15
U = matread("basis.mat")["U"][:, 1:Ncoef]

model_biased = BSON.load("msecrb_network.bson")[:model]
model_unbiased = BSON.load("biasreduced_network.bson")[:model]

## simulation parameters
m0s = 0.2
R1f = 1/1.92
R2f = 1/77.6e-3
Rex  = 16.5
R1s = 1/0.337
T2s = 12.4e-6
ω0 = 0
B1  = 1

σ = 1/20
Nr = 1000 # number of noise realizations

## uncomment the parameter of interest
# ps = :m0s; p = 0:0.05:0.4
# ps = :R1f; p = 0.2:0.1:1; pstring=L"R_1^f~~(1/s)"
ps = :R2f; p = 5:2:20; pstring=L"R_2^f~~(1/s)"
# ps = :Rex; p = 10:2:25
# ps = :R1s; p = 2:0.2:4; pstring=L"R_1^s~~(1/s)"
# ps = :T2s; p = 5e-6:1e-6:15e-6
# ps = :B1; p = 0.6:0.1:1.3

## compute NN estimates
m0s_vals = zeros(2, Nr, length(p))
R1f_vals = zeros(2, Nr, length(p))
R2f_vals = zeros(2, Nr, length(p))
Rex_vals = zeros(2, Nr, length(p))
R1s_vals = zeros(2, Nr, length(p))
T2s_vals = zeros(2, Nr, length(p))
pvals = zeros(6, length(p))
for ip ∈ eachindex(p)
    eval(:($ps = p[$ip]))

    s = Array{ComplexF64}(undef, length(α[1]), length(α))
    s .= 0
    Threads.@threads for is ∈ eachindex(α)
        s[:,is,:] .= dropdims(calculatesignal_linearapprox(α[is], TRF[is], TR, ω0, B1, m0s, R1f, R2f, Rex, R1s, T2s, R2slT), dims=2)
    end

    x = repeat(vec(s), 1, Nr)
    x .+= σ .* (randn(size(x)) .+ 1im * randn(size(x)))
    x = ComplexF32.(transpose(U' * x))

    m0s_vals[1,:,ip], R1f_vals[1,:,ip], R2f_vals[1,:,ip], Rex_vals[1,:,ip], R1s_vals[1,:,ip], T2s_vals[1,:,ip] = fit_invivo(model_biased, x, trues(size(x,1)))
    m0s_vals[2,:,ip], R1f_vals[2,:,ip], R2f_vals[2,:,ip], Rex_vals[2,:,ip], R1s_vals[2,:,ip], T2s_vals[2,:,ip] = fit_invivo(model_unbiased, x, trues(size(x,1)))

    pvals[1,ip] = pvalue(UnequalVarianceTTest(m0s_vals[1,:,ip], m0s_vals[2,:,ip]))
    pvals[2,ip] = pvalue(UnequalVarianceTTest(R1f_vals[1,:,ip], R1f_vals[2,:,ip]))
    pvals[3,ip] = pvalue(UnequalVarianceTTest(R2f_vals[1,:,ip], R2f_vals[2,:,ip]))
    pvals[4,ip] = pvalue(UnequalVarianceTTest(Rex_vals[1,:,ip], Rex_vals[2,:,ip]))
    pvals[5,ip] = pvalue(UnequalVarianceTTest(R1s_vals[1,:,ip], R1s_vals[2,:,ip]))
    pvals[6,ip] = pvalue(UnequalVarianceTTest(T2s_vals[1,:,ip], T2s_vals[2,:,ip]))
end

## plot areas of significant bias reduction
pvals_plot = (pvals .<= 0.05 / length(pvals))
pvals_plot[1,:] .= pvals_plot[1,:] .&& vec(diff(abs.(mean(m0s_vals, dims=2) .- m0s), dims=1) .< 0)
pvals_plot[2,:] .= pvals_plot[2,:] .&& vec(diff(abs.(mean(R1f_vals, dims=2) .- R1f), dims=1) .< 0)
pvals_plot[3,:] .= pvals_plot[3,:] .&& vec(diff(abs.(mean(R2f_vals, dims=2) .- R2f), dims=1) .< 0)
pvals_plot[4,:] .= pvals_plot[4,:] .&& vec(diff(abs.(mean(Rex_vals, dims=2) .- Rex), dims=1) .< 0)
pvals_plot[5,:] .= pvals_plot[5,:] .&& vec(diff(abs.(mean(R1s_vals, dims=2) .- R1s), dims=1) .< 0)
pvals_plot[6,:] .= pvals_plot[6,:] .&& vec(diff(abs.(mean(T2s_vals, dims=2) .- T2s), dims=1) .< 0)
pvals_plot = pvals_plot[[1,2,4,5,6],:]

plot(heatmap(5:2:20, 1:5, pvals_plot), yflip=true, colorbar=false, tick_direction=:out,
	yticks=(1:5, [L"m_0^s" L"R_1^f" L"R_\mathrm{x}" L"R_1^s" L"T_2^s"]), xlabel=L"R_2^f~~(1/s)", xticks=p,
	ytickfontsize=20, xtickfontsize=12, xguidefontsize=20, size=(500,500), dpi=300)

## boxplots
eval(:($ps = p))
x_ticks = [5,7,11,15,19]
pm0s  = plot(p[[1,end]], [m0s[1], m0s[end]],        xlabel = "",      xticks=(x_ticks, ""), seriescolor=cur_colors[2], xlim=(4,22.5), ylim=(0.11,0.27), yticks =0.11:0.04:0.27, ylabel = L"m_0^s", legend=:none);
pm0s2 = plot(p[[1,end]], [m0s[1], m0s[end]],        xlabel = "",      xticks=(x_ticks, ""), seriescolor=cur_colors[2], xlim=(4,22.5), ylim=(0.11,0.27), yticks=(0.11:0.04:0.27, ""), legend=:none);
pR1f  = plot(p[[1,end]], [R1f[1], R1f[end]],        xlabel = "",      xticks=(x_ticks, ""), seriescolor=cur_colors[2], xlim=(4,22.5), ylim=(0.25,0.7), yticks= 0.25:0.15:0.7, ylabel = L"R_1^f~~(1/s)", legend=:none);
pR1f2 = plot(p[[1,end]], [R1f[1], R1f[end]],        xlabel = "",      xticks=(x_ticks, ""), seriescolor=cur_colors[2], xlim=(4,22.5), ylim=(0.25,0.7), yticks=(0.25:0.15:0.7, ""), legend=:none);
pR2f  = plot(p[[1,end]], [R2f[1], R2f[end]],        xlabel = "",      xticks=(x_ticks, ""), seriescolor=cur_colors[2], xlim=(4,22.5), ylim=(4,22.5), yticks= [5,7,11,15,19], ylabel = L"R_2^f~~(1/s)", legend=:none, aspect_ratio=1);
pR2f2 = plot(p[[1,end]], [R2f[1], R2f[end]],        xlabel = "",      xticks=(x_ticks, ""), seriescolor=cur_colors[2], xlim=(4,22.5), ylim=(4,22.5), yticks=([5,7,11,15,19], ""), legend=:none);
pRex  = plot(p[[1,end]], [Rex[1], Rex[end]],        xlabel = pstring, xticks= x_ticks,       seriescolor=cur_colors[2], xlim=(4,22.5), ylim=(6,30), yticks =6:4:30, ylabel = L"R_x~~(1/s)", legend=:none);
pRex2 = plot(p[[1,end]], [Rex[1], Rex[end]],        xlabel = pstring, xticks= x_ticks,       seriescolor=cur_colors[2], xlim=(4,22.5), ylim=(6,30), yticks=(6:4:30, ""), legend=:none);
pR1s = plot(p[[1,end]], [R1s[1], R1s[end]],         xlabel = pstring, xticks= x_ticks,       seriescolor=cur_colors[2], xlim=(4,22.5), ylim=(1.3,5.7), yticks =1:1:6, ylabel = L"R_1^s~~(1/s)", legend=:none);
pR1s2 = plot(p[[1,end]], [R1s[1], R1s[end]],        xlabel = pstring, xticks= x_ticks,       seriescolor=cur_colors[2], xlim=(4,22.5), ylim=(1.3,5.7), yticks=(1:1:6, ""), legend=:none);
pT2s  = plot(p[[1,end]], [T2s[1], T2s[end]] .* 1e6, xlabel = pstring, xticks= x_ticks,       seriescolor=cur_colors[2], xlim=(4,22.5), ylim=(4,22), yticks=5:5:20, ylabel = L"T_2^s~~(μs)", legend=:none);
pT2s2 = plot(p[[1,end]], [T2s[1], T2s[end]] .* 1e6, xlabel = pstring, xticks= x_ticks,       seriescolor=cur_colors[2], xlim=(4,22.5), ylim=(4,22), yticks=(5:5:20, ""), legend=:none);
 
for ip ∈ eachindex(p)
    eval(:($ps = p[$ip]))

    boxplot!(pm0s,  p[ip] .* ones(Nr), m0s_vals[1,:,ip]       , seriescolor=cur_colors[1], outliers=false, bar_width = (p[end]-p[1])/length(p)/2)
    boxplot!(pR1f,  p[ip] .* ones(Nr), R1f_vals[1,:,ip]       , seriescolor=cur_colors[1], outliers=false, bar_width = (p[end]-p[1])/length(p)/2)
    boxplot!(pR2f,  p[ip] .* ones(Nr), R2f_vals[1,:,ip]       , seriescolor=cur_colors[1], outliers=false, bar_width = (p[end]-p[1])/length(p)/2)
    boxplot!(pRex,  p[ip] .* ones(Nr), Rex_vals[1,:,ip]       , seriescolor=cur_colors[1], outliers=false, bar_width = (p[end]-p[1])/length(p)/2)
    boxplot!(pR1s,  p[ip] .* ones(Nr), R1s_vals[1,:,ip]       , seriescolor=cur_colors[1], outliers=false, bar_width = (p[end]-p[1])/length(p)/2)
    boxplot!(pT2s,  p[ip] .* ones(Nr), T2s_vals[1,:,ip] .* 1e6, seriescolor=cur_colors[1], outliers=false, bar_width = (p[end]-p[1])/length(p)/2)
    boxplot!(pm0s2, p[ip] .* ones(Nr), m0s_vals[2,:,ip]       , seriescolor=cur_colors[1], outliers=false, bar_width = (p[end]-p[1])/length(p)/2)
    boxplot!(pR1f2, p[ip] .* ones(Nr), R1f_vals[2,:,ip]       , seriescolor=cur_colors[1], outliers=false, bar_width = (p[end]-p[1])/length(p)/2)
    boxplot!(pR2f2, p[ip] .* ones(Nr), R2f_vals[2,:,ip]       , seriescolor=cur_colors[1], outliers=false, bar_width = (p[end]-p[1])/length(p)/2)
    boxplot!(pRex2, p[ip] .* ones(Nr), Rex_vals[2,:,ip]       , seriescolor=cur_colors[1], outliers=false, bar_width = (p[end]-p[1])/length(p)/2)
    boxplot!(pR1s2, p[ip] .* ones(Nr), R1s_vals[2,:,ip]       , seriescolor=cur_colors[1], outliers=false, bar_width = (p[end]-p[1])/length(p)/2)
    boxplot!(pT2s2, p[ip] .* ones(Nr), T2s_vals[2,:,ip] .* 1e6, seriescolor=cur_colors[1], outliers=false, bar_width = (p[end]-p[1])/length(p)/2)
end

plot(pR2f, pR2f2, pm0s, pm0s2, pR1f, pR1f2, pRex, pRex2, pR1s, pR1s2, pT2s, pT2s2, aspect_ratio=:auto,
    layout=(2,6), size=(1300,400), bottom_margin=10mm, left_margin=10mm, yguidefontsize=16, xguidefontsize=16, xtickfontsize=10, ytickfontsize=10, dpi=300)