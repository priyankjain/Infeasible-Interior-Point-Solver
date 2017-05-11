type IplpSolution
  x::Vector{Float64} # the solution vector
  flag::Bool         # a true/false flag indicating convergence or not
  cs::Vector{Float64} # the objective vector in standard form
  As::SparseMatrixCSC{Float64} # the constraint matrix in standard form
  bs::Vector{Float64} # the right hand side (b) in standard form
  xs::Vector{Float64} # the solution in standard form
  lam::Vector{Float64} # the solution lambda in standard form
  s::Vector{Float64} # the solution s in standard form
end

type IplpProblem
  c::Vector{Float64}
  A::SparseMatrixCSC{Float64}
  b::Vector{Float64}
  lo::Vector{Float64}
  hi::Vector{Float64}
end

function convert_matrixdepot(mmmeta::Dict{AbstractString,Any})
  key_base = sort(collect(keys(mmmeta)))[1]
  return IplpProblem(
    vec(mmmeta[key_base*"_c"]),
    mmmeta[key_base],
    vec(mmmeta[key_base*"_b"]),
    vec(mmmeta[key_base*"_lo"]),
    vec(mmmeta[key_base*"_hi"]))
end


using MatrixDepot
root = "LPnetlib";
names = ["lp_afiro", "lp_brandy", "lp_fit1d", "lp_adlittle", "lp_agg", "lp_ganges", "lp_stocfor1", "lp_25fv47", "lpi_chemcom"]

using MathProgBase, Clp
include("project2.jl")

k = 0;
for name in names
    k = k + 1;
    println("=======================================>")
    println("Problem $(k): $(name)")
    D = 0;
    try
        D = matrixdepot(joinpath(root,name), :read, meta=true)
    catch
        matrixdepot(joinpath(root,name), :get)
        D = matrixdepot(joinpath(root,name), :read, meta=true)
    end
    P = convert_matrixdepot(D)

    # using standard solver
    println("\nClp solver:")
    tic();
    sol = linprog(P.c, P.A, '=', P.b, P.lo, P.hi, ClpSolver());
    println("Solution status: $(sol.status)")
    println("Optimal objective value: $(sol.objval)")
    toc()

    println("\nOur solver:")
    tic();
    S = iplp(P, 1e-8, maxit=100);
    objval = dot(P.c, S.x);
    println("Optimality status: $(S.flag)")
    println("Optimal objective value: $(objval)")
    toc()

    println(" ")
end
