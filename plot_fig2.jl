using Pkg
Pkg.activate(".")
Pkg.instantiate()

include("NN_functions.jl")
using MAT, BSON
using MRIgeneralizedBloch
using LinearAlgebra, Statistics, HypothesisTests
using LaTeXStrings, Measures
using Plots
default(markerstrokecolor=:auto)

## load flip angle pattern
TR = 3.5e-3
control = matread("control.mat")
α = [reshape(control["alpha"],:,6)[:,i] for i = 1:size(reshape(control["alpha"],:,6),2)]
TRF = [reshape(control["TRF"],:,6)[:,i] for i = 1:size(reshape(control["TRF"],:,6),2)]
R2slT = precompute_R2sl(T2s_min=12e-6, T2s_max=13e-6, B1_max=1)

## load basis
Ncoef = 15
U = matread("basis.mat")["U"][:,1:Ncoef]

## simulate canonical WM fingerprint
m0s = 0.2
R1f = 1/1.92
R2f = 1/77.6e-3
Rx  = 16.5
R1s = 1/0.337
T2s = 12.4e-6
truth = [m0s, R1f, R2f, Rx, R1s, T2s]

s = zeros(ComplexF32, length(α[1]), length(α), 9) # 9 model parameters
Threads.@threads for j ∈ eachindex(α)
    si = calculatesignal_linearapprox(α[j], TRF[j], TR, 0, 1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT;
        grad_list=[grad_m0s(), grad_R1f(), grad_R2f(), grad_Rx(), grad_R1s(), grad_T2s(), grad_ω0(), grad_B1()])
    s[:,j,:] .= dropdims(si, dims=2)
end
s = reshape(s, :, 9)
CRB = real.(diag(inv(s' * s)))[2:1+length(truth)]
sc = U' * s
CCRB = real.(diag(inv(sc' * sc)))[2:1+length(truth)]
sc = sc[:,1]

## generate noise corrupted samples
# note these will be slightly different than those used for the manuscript's figures
Nr = 1000
SNR = 10:50
scn = zeros(Float32, Ncoef*2, Nr, length(SNR))
for i ∈ eachindex(SNR)
    scn[:,:,i] .= dropdims(preprocess_fingerprints(reshape(sc, Ncoef, 1); SNR_min=SNR[i], SNR_max=SNR[i], Nr=Nr), dims=2)
end

## Load neural networks
model = Vector{Any}(undef, 0)
push!(model, BSON.load("msecrb_network.bson")[:model])
push!(model, BSON.load("biasreduced_network.bson")[:model])

## NN fitting
Np = 6 # how many parameters the NNs are trained to output
p = zeros(Float32, Np, Nr, length(SNR), length(model))
@time for i ∈ axes(model,1)
    p[:,:,:,i] .= reshape(convert_params(model[i](reshape(scn, Ncoef*2, :)))[1:Np,:], Np, Nr, length(SNR))
end

## bias/variance calculation
bias = 100 .* abs.(dropdims(mean(p, dims=2), dims=2) .- truth) ./ truth
variance = dropdims(var(p; dims=2), dims=2)

function plot_biasvariance()
	gr()
	cur_colors = theme_palette(:auto)

	ylim1 = [-0.3,25]; yticks1 = (0:5:25);
	ylim2 = [-0.3,25]; yticks2 = (0:5:25);
	ylim3 = [-0.2,12]; yticks3 = (0:4:12);
	ylim4 = [-0.5,45]; yticks4 = (0:10:50);
	ylim5 = [-0.5,40]; yticks5 = (0:10:40);
	ylim6 = [-0.8,60]; yticks6 = (0:15:60);
	
	pm0s = plot(    SNR, zeros(length(SNR)), seriescolor=:black, linewidth=5, legend=:topright, label=" Efficient")
	scatter!(pm0s,  SNR, bias[1,:,1], markersize=3.5, markershape=:circle, seriescolor=cur_colors[1], label=" MSE-CRB", title=L"~|\mathrm{bias}~\%|", ylabel=L"m_0^s")
	scatter!(pm0s,  SNR, bias[1,:,2], markersize=3.5, markershape=:circle, seriescolor=cur_colors[2], label=" Bias-Reduced", xticks=(10:10:50,""), ylim=ylim1, yticks=yticks1, legend_rows=2)
	pm0s2 = plot(   SNR, 100/truth[1] .* sqrt.(CCRB[1]) ./ SNR, label="", seriescolor=:black, linestyle=:solid, linewidth=5, ylim=ylim1)
	scatter!(pm0s2, SNR, 100/truth[1] .* sqrt.(variance[1,:,1]), markersize=4, markershape=:diamond, seriescolor=cur_colors[1], label="", title=L"~\mathrm{std}~\%")
	scatter!(pm0s2, SNR, 100/truth[1] .* sqrt.(variance[1,:,2]), markersize=4, markershape=:diamond, seriescolor=cur_colors[2], label="", yticks=(yticks1, ""), xticks=(10:10:50,""))

	pR1f = plot(    SNR, zeros(length(SNR)), seriescolor=:black, linewidth=5, label="")
	scatter!(pR1f,  SNR, bias[2,:,1], markersize=3.5, markershape=:circle, seriescolor=cur_colors[1], label="", ylabel=L"R_1^f", xticks=(10:10:50, ""))
	scatter!(pR1f,  SNR, bias[2,:,2], markersize=3.5, markershape=:circle, seriescolor=cur_colors[2], label="", ylim=ylim2, yticks=yticks2)
	pR1f2 = plot(   SNR, 100/truth[2] .* sqrt.(CCRB[2]) ./ SNR, label="", seriescolor=:black, linestyle=:solid, linewidth=5)
	scatter!(pR1f2, SNR, 100/truth[2] .* sqrt.(variance[2,:,1]), markersize=4, markershape=:diamond, seriescolor=cur_colors[1], label="", ylim=ylim2, yticks=(yticks2, ""))
	scatter!(pR1f2, SNR, 100/truth[2] .* sqrt.(variance[2,:,2]), markersize=4, markershape=:diamond, seriescolor=cur_colors[2], label="", xticks=(10:10:50,""))

	pR2f = plot(    SNR, zeros(length(SNR)), seriescolor=:black, linewidth=5, label="")
	scatter!(pR2f,  SNR, bias[3,:,1], markersize=3.5, markershape=:circle, seriescolor=cur_colors[1], label="", ylabel=L"R_2^f", xticks=(10:10:50,""))
	scatter!(pR2f,  SNR, bias[3,:,2], markersize=3.5, markershape=:circle, seriescolor=cur_colors[2], label="", ylim=ylim3, yticks=yticks3)
	pR2f2 = plot(   SNR, 100/truth[3] .* sqrt.(CCRB[3]) ./ SNR, label="", seriescolor=:black, linestyle=:solid, linewidth=5)
	scatter!(pR2f2, SNR, 100/truth[3] .* sqrt.(variance[3,:,1]), markersize=4, markershape=:diamond, seriescolor=cur_colors[1], label="", ylim=ylim3, yticks=(yticks3, ""))
	scatter!(pR2f2, SNR, 100/truth[3] .* sqrt.(variance[3,:,2]), markersize=4, markershape=:diamond, seriescolor=cur_colors[2], label="", xticks=(10:10:50,""))

	pRx = plot(    SNR, zeros(length(SNR)), seriescolor=:black, linewidth=5, label="")
	scatter!(pRx,  SNR, bias[4,:,1], markersize=3.5, markershape=:circle, seriescolor=cur_colors[1], label="", title=L"~|\mathrm{bias}~\%|", ylabel=L"R_\mathrm{x}")
	scatter!(pRx,  SNR, bias[4,:,2], markersize=3.5, markershape=:circle, seriescolor=cur_colors[2], label="", ylim=ylim4, yticks=yticks4, xticks=(10:10:50,""))
	pRx2 = plot(   SNR, 100/truth[4] .* sqrt.(CCRB[4]) ./ SNR, label="", seriescolor=:black, linestyle=:solid, linewidth=5)
	scatter!(pRx2, SNR, 100/truth[4] .* sqrt.(variance[4,:,1]), markersize=4, markershape=:diamond, seriescolor=cur_colors[1], label="", ylim=ylim4, yticks=(yticks4, ""))
	scatter!(pRx2, SNR, 100/truth[4] .* sqrt.(variance[4,:,2]), markersize=4, markershape=:diamond, seriescolor=cur_colors[2], label="", xticks=(10:10:50,""), title=L"~\mathrm{std}~\%")

	pR1s = plot(    SNR, zeros(length(SNR)), seriescolor=:black, linewidth=5, label="")
	scatter!(pR1s,  SNR, bias[5,:,1], markersize=3.5, markershape=:circle, seriescolor=cur_colors[1], label="", ylabel=L"R_1^s")
	scatter!(pR1s,  SNR, bias[5,:,2], markersize=3.5, markershape=:circle, seriescolor=cur_colors[2], label="", ylim=ylim5, yticks=yticks5, xlabel=L"|M_0|/σ")
	pR1s2 = plot(   SNR, 100/truth[5] .* sqrt.(CCRB[5]) ./ SNR, label="", seriescolor=:black, linestyle=:solid, linewidth=5)
	scatter!(pR1s2, SNR, 100/truth[5] .* sqrt.(variance[5,:,1]), markersize=4, markershape=:diamond, seriescolor=cur_colors[1], label="", ylim=ylim5, yticks=(yticks5,""))
	scatter!(pR1s2, SNR, 100/truth[5] .* sqrt.(variance[5,:,2]), markersize=4, markershape=:diamond, seriescolor=cur_colors[2], label="", xlabel=L"|M_0|/σ")

	pT2s = plot(    SNR, zeros(length(SNR)), seriescolor=:black, linewidth=5, label="")
	scatter!(pT2s,  SNR, bias[6,:,1], markersize=3.5, markershape=:circle, seriescolor=cur_colors[1], label="", ylabel=L"T_2^s")
	scatter!(pT2s,  SNR, bias[6,:,2], markersize=3.5, markershape=:circle, seriescolor=cur_colors[2], label="", ylim=ylim6, yticks=yticks6, xlabel=L"|M_0|/σ", xticks=0:10:50)
	pT2s2 = plot(   SNR, 100/truth[6] .* sqrt.(CCRB[6]) ./ SNR, label="", seriescolor=:black, linestyle=:solid, linewidth=5)
	scatter!(pT2s2, SNR, 100/truth[6] .* sqrt.(variance[6,:,1]), markersize=4, markershape=:diamond, seriescolor=cur_colors[1], label="", ylim=ylim6, yticks=(yticks6, ""))
	scatter!(pT2s2, SNR, 100/truth[6] .* sqrt.(variance[6,:,2]), markersize=4, markershape=:diamond, seriescolor=cur_colors[2], label="", xlabel=L"|M_0|/σ")

	p = plot(pm0s, pm0s2, pRx, pRx2, pR1f, pR1f2, pR2f, pR2f2, pR1s, pR1s2, pT2s, pT2s2, layout=(3,4), size=(1400,850), dpi=300, markerstrokewidth=2,
		legend=:topright, yguidefontsize=24, xguidefontsize=20, titlefontsize=24, xtickfontsize=12, ytickfontsize=12, legendfontsize=16, tick_direction=:out,
		left_margin=8mm, bottom_margin=5mm)
	return p
end

plot_biasvariance()

## statistical analysis
pvals = zeros(6, 41)
for i ∈ CartesianIndices(pvals)
	pvals[i] = pvalue(UnequalVarianceTTest(p[i[1],:,i[2],1], p[i[1],:,i[2],2]))
end
plot(heatmap(SNR, 1:6, (pvals .<= 0.05 / length(pvals)) .&& (bias[:,:,2] .<= bias[:,:,1])), yflip=true, colorbar=false, tick_direction=:out,
	yticks=(1:6, [L"m_0^s" L"R_1^f" L"R_2^f" L"R_\mathrm{x}" L"R_1^s" L"T_2^s"]), xlabel=L"|M_0|/σ",
	ytickfontsize=20, xtickfontsize=12, xguidefontsize=20, size=(500,500), dpi=300)