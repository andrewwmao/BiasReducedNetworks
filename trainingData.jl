
using Pkg
Pkg.activate(".")
Pkg.instantiate()
using LinearAlgebra
using MAT

## User-Defined Parameters
files = 1:40
TR = 3.5e-3
R = 15

## merge fingerprints into td.mat using a precomputed basis
U = matread("basis.mat")["U"][:,1:R]
Nfiles = length(files)
(Np, Nbatch) = size(matread("fingerprints/ijob$(files[1]).mat")["p"])
sc   = zeros(ComplexF32, R, Nbatch, Nfiles) # compressed fingerprints
p    = zeros(Float64, Np, Nbatch, Nfiles)   # ground-truth parameters
CRBc = zeros(Float64, Np, Nbatch, Nfiles)   # Compressed Cramér-Rao Bound
@time for i ∈ eachindex(files)
    data_file = "fingerprints/ijob$(files[i]).mat"
    try
        file = matopen(data_file)
        s = read(file, "s")
        p[:,:,i] .= read(file, "p")
        close(file)
        for j ∈ axes(s,3)
            s_temp = U' * @view(s[:,:,j]) # calculate compressed CRB
            sc[:,j,i] .= s_temp[:,1]
            D = zeros(Int16, size(s_temp,2))
            FIM = s_temp' * s_temp
            for k ∈ axes(p,1)
                D[1+k] = 1
                CRB = FIM \ D
                CRBc[k,j,i] = real(CRB[1+k])
                D[1+k] = 0
            end
        end
    catch
        println("ijob$(i) does not exist")
    end
    GC.gc()
end
sc   = reshape(sc, R, :)
p    = reshape(p, Np, :)
CRBc = reshape(CRBc, size(p,1), :)
p[7,:] .= abs.(p[7,:]) ./ (π/TR) #normalize range of w0 to [-1,1]
CRBc[7,:] ./= (π/TR)^2

## correct for missing jobs
idx_nonzero = findall(!iszero, p[2,:])
if length(idx_nonzero) < size(sc,2)
    sc   = sc[:, idx_nonzero]
    p    = p[:, idx_nonzero]
    CRBc = CRBc[:, idx_nonzero]
    GC.gc()
end

## Save training dataset
file = matopen("td.mat", "w");
write(file, "sc", sc);
write(file, "p", p);
write(file, "CRB", CRBc);
close(file);