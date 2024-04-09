# Bias Reduced Neural Networks for Parameter Estimation in Quantitative MRi
This repository aims to partially reproduce the results in the paper https://arxiv.org/abs/2312.11468. These neural networks have reduced bias compared to those trained with the typical mean-squared error loss, and a variance that is close to the Cramér-Rao Bound.

```sim.jl``` and ```sim.sh``` are an example Julia script to calculate the training data using the [MRIgeneralizedBloch.jl](https://github.com/JakobAsslaender/MRIgeneralizedBloch.jl) package, and a corresponding bash script to submit it as a job to a computational cluster managed by Slurm.

```td.jl``` and ```td.sh``` are used to calculate the training data, ```td.mat```, from the simulated fingerprints, compressed to a low-rank space. In this case, a precomputed basis ```basis.mat``` is provided, calculated using the method described in the paper https://arxiv.org/abs/2305.00326 and by the corresponding repository https://github.com/andrewwmao/CRBBasis.

```train.jl``` and ```train.sh``` are example scripts for training the Bias-Reduced networks.

```plot_fig1.jl```, ```plot_fig2.jl```, and ```plot_fig4.jl``` partially reproduce Figs. 1, 2, and 4 in the Bias-Reduced networks paper. The parts that require non-linear least squares fitting can be reproduced using code already provided in [MRIgeneralizedBloch.jl](https://github.com/JakobAsslaender/MRIgeneralizedBloch.jl).

Our preprint is currently under revision at Magnetic Resonance in Medicine. This repository will be finalized when the paper is published.
