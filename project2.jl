"""
soln = iplp(Problem,tol) solves the linear program:

minimize c'*x where Ax = b and lo <= x <= hi

where the variables are stored in the following struct:

Problem.A
Problem.c
Problem.b   
Problem.lo
Problem.hi

and the IplpSolution contains fields 

[x,flag,cs,As,bs,xs,lam,s]

which are interpreted as   
a flag indicating whether or not the
solution succeeded (flag = true => success and flag = false => failure),

along with the solution for the problem converted to standard form (xs):

minimize cs'*xs where As*xs = bs and 0 <= xs

and the associated Lagrange multipliers (lam, s).

This solves the problem up to 
the duality measure (xs'*s)/n <= tol and the normalized residual
norm([As'*lam + s - cs; As*xs - bs; xs.*s])/norm([bs;cs]) <= tol
and fails if this takes more than maxit iterations.
"""

#######################################
# The interfacing function
function iplp(Problem, tol; maxit=100)
    Solution, rmCols, lo, splitVar = presolve(Problem);
    Solution = solve(Solution, tol, maxit);
    Solution = postsolve(Solution, rmCols, lo, splitVar); 
    return Solution
end

#######################################
# a working presolver
# 0) split -inf bounds
# 1) convert standard form
# 2) check bounds
# 3) remove empty and singleton rows
# 4) remove empty columns
# 5) remove duplicate rows
function presolve(Problem)
    eps = 1e-10;
    inf = 1e+10;
    A = Problem.A;
    c = Problem.c;
    b = Problem.b;  
    lo = Problem.lo;
    hi = Problem.hi;
    println("Problem size: $(size(Problem.A))")

    # check unbounded variables
    # split unbounded variables to x = (x+) - (x-)
    splitVar = [];
    n = size(A,2);
    for i = 1:n
        if lo[i] <= -inf
            c = vcat(c, -c[i]);
            A = hcat(A, -A[:,i]);          
            lo = vcat(lo, 0.0);
            hi = vcat(hi, abs(lo[i]));
            lo[i] = 0.0
            push!(splitVar, (i, length(lo)));
        end
    end

    # remove lower bounds
    # NOTE: postsolve x += lo
    hi -= lo;
    b -= A * lo;

    m, n = size(A);
    x0 = zeros(n); # presolved fixed primals
    rmCols = zeros(Bool, n); # variables
    rmRows = zeros(Bool, m); # constraints

    # add slacks to remove bounnds ==> standard form
    num_slack = 0;
    for i = 1:n
        if hi[i] < 0 # lo > hi
            error("Infeasible bounds (lo > hi) ===> Program stops.")
        elseif hi[i] <= eps # fixed bound lo = x = hi
            #println("Found fixed bound @var $(i)")
            x0[i] = lo[i];
            rmCols[i] = true;
            b -= A[:,i] * x0[i];
        elseif hi[i] < inf # x is bounded => add to constraint
            b = vcat(b, hi[i]) # the upper bound
            A = hcat(A, zeros(size(A,1))); # augment A
            A = vcat(A, zeros(1, size(A,2)));
            A[end, i] = 1.0; # only the last row should change
            A[end, end] = 1.0; # doubleton constraint
            num_slack += 1;
        end
    end
    println("Added $(num_slack) slack variables.")
    m += num_slack;
    n += num_slack;

    c_aug = vcat(c, zeros(num_slack));
    x0 = vcat(x0, zeros(num_slack));
    rmCols = vcat(rmCols, zeros(Bool, num_slack));
    rmRows = vcat(rmRows, zeros(Bool, num_slack));

    # reduce problem
    repeat = true;
    while repeat
        repeat = false;

        # check empty and singleton rows
        for i = 1:m
            if rmRows[i]; continue; end
            idx = [j for j = 1:n if ~rmCols[j] && abs(A[i,j]) > eps];
            if length(idx) == 0 # empty row
                #println("Found empty row @ $(i)")
                rmRows[i] = true; repeat = true;
                #if abs(b[i]) > 0.0; error("Infeasible constraint ===> Program stops."); end
            elseif length(idx) == 1 # singleton row
                #println("Found singleton row @ $(i)")
                j = idx[1]; rmCols[j] = true;
                x0[j] = b[i] / A[i,j];
                b[~rmRows] -= A[~rmRows,j] * x0[j];
                rmRows[i] = true; repeat = true;
            end
        end

        # check empty cols
        for j = 1:n
            if rmCols[j]; continue; end
            idx = [i for i = 1:m if ~rmRows[i] && abs(A[i,j]) > eps];
            if length(idx) == 0 # empty col
                #println("Found empty col @ $(j)")
                rmCols[j] = true; repeat = true;
                # check boundedness
                if c_aug[j] > 0
                    x0[j] = 0.0;
                elseif c_aug[j] == 0
                    x0[j] = -lo[j];
                elseif c_aug[j] < 0 && hi[j] < inf
                    x0[j] = hi[j];
                elseif c_aug[j] < 0 && hi[j] >= inf
                    error("Unbounded problem. ===> Program stops.")
                end
            elseif length(idx) == 1 # singleton col
                continue;
            end
        end

        # duplicate rows
        for i = 1:m # basis pattern
            if rmRows[i]; continue; end
            x = (abs(A[i,:]).>eps) & ~rmCols;
            if ~any(x); continue; end
            for ii = i+1:m # substrat i-th row from this row
                if rmRows[ii]; continue; end
                y = (abs(A[ii,:]).>eps) & ~rmCols;
                if all(x .== y) # same pattern
                    repeat = true;
                    k = minimum(A[ii,y] ./ A[i,x]);
                    A[ii,~rmCols] -= k * A[i,~rmCols];
                    b[ii] -= k * b[i];
                end
            end
        end
    end

    c_aug = c_aug[~rmCols];
    A = A[~rmRows, ~rmCols];
    b = b[~rmRows];

    # minimum norm least square initialization
    AAT = A*A';
    I = 1e-6 * eye(AAT);
    x = A' * ((I+AAT) \ b);
    lam = (I+AAT) \ (A * c_aug);
    s = c_aug - A' * lam;

    # ensure non-negativity
    dx = max(-3/2 * minimum(x), 0)
    ds = max(-3/2 * minimum(s), 0)
    x = x .+ dx
    s = s .+ ds

    println("Presolved problem size: $(size(A))")
    Solution = IplpSolution(x0, false, c_aug, A, b, x, lam, s)
    return (Solution, rmCols, lo, splitVar)
end

#############################################
# postsolve following presolve
# 1) add the lower bound back
# 2) merge any split variable
function postsolve(Solution, rmCols, lo, splitVar)
    x = Solution.x;

    x[~rmCols] = Solution.xs;
    x[1:length(lo)] += lo;
    
    if length(splitVar) > 0
        for k = splitVar
            x[k[1]] -= x[k[2]];
        end
    end
    
    n = length(lo) - length(splitVar);
    Solution.x = x[1:n];
    return Solution
end

#############################################
# the main solver function
function solve(Solution, tol, maxit)
    # the problem state
    A = Solution.As;
    c = Solution.cs;
    b = Solution.bs;

    # the variables
    x = Solution.xs;
    s = Solution.s;
    lam = Solution.lam;

    m, n = size(A);

    # KKT criteria
    KKTs(x, lam, s) = norm([A'*lam + s - c; A*x - b; x.*s])/norm([b;c])
    Duality(x, s) = dot(x,s)/n;

    for iter = 1:maxit
        kkt = KKTs(x, lam, s);
        gap = Duality(x, s);

        if kkt <= tol && gap <= tol
            Solution.xs = x;
            Solution.lam = lam;
            Solution.s = s;
            Solution.flag = true;
            println("Solution converged in $(iter) iterations.")
            return Solution
        elseif gap >= 1e64
            Solution.xs = x;
            Solution.lam = lam;
            Solution.s = s;
            Solution.flag = false;
            println("Duality gap diverges.")
            return Solution
        end

        # precompute matrices
        X = diagm(x);
        S = diagm(s);
        E = ones(n);

        #Sinv = inv(S);
        D = S \ X;            
        ADAT = A * D * A';

        # residuals (Infeasible IP method)
        rb = b - A * x;
        rc = c - A' * lam - s;
        #r = - X * S * E;

        # factorize H
        L = CholFact(ADAT);

        # affine step (predictor)
        dlam = L' \ (L \ (rb + A * x + A * D * rc));
        ds = rc - A' * dlam;
        dx = - x - D * ds;

        # calculate step size (affine)
        lr_p, lr_d = step_size(x, s, dx, ds);
        lr_p = min(1.0, lr_p);
        lr_d = min(1.0, lr_d);

        # update affine
        x_aff = x + lr_p * dx;
        lam_aff = lam + lr_d * dlam;
        s_aff = s + lr_d * ds;

        # Mehrotra's corrector
        sigma = (Duality(x_aff, s_aff) / gap) ^ 3;

        # solve corretor system
        r = diagm(dx) * diagm(ds) * E - sigma * gap * E;
        dlam_cc = L' \ (L \ (A / S * r));
        ds_cc = - A' * dlam_cc;
        dx_cc = - D * ds_cc - S \ r;

        # new direction
        dx += dx_cc;
        dlam += dlam_cc;
        ds += ds_cc;

        # step size (corrector)
        lr_p, lr_d = step_size(x, s, dx, ds);
        lr_p = min(0.99 * lr_p, 1.0);
        lr_d = min(0.99 * lr_d, 1.0);
        
        # update variable after correction
        x += lr_p * dx;
        lam += lr_d * dlam;
        s += lr_d * ds;
    end

    println("Exceed maximum iteration.")
    Solution.xs = x;
    Solution.s = s;
    Solution.lam = lam;
    return Solution
end

###############################################
# compute step size
function step_size(x, s, dx, ds)
    lr_p = Inf;
    lr_d = Inf;

    # check out the negative direction
    # ensure x >= 0 and s >= 0
    for i = 1:length(x)
        if dx[i] < 0
            lr_p = min(lr_p,  -x[i] / dx[i]);
        end
        if ds[i] < 0
            lr_d = min(lr_d,  -s[i] / ds[i]);
        end
    end
    return (lr_p, lr_d)
end

################################################
# Cholesky factorization
# replace small pivots
function CholFact(M)
    # Cholesky factorization
    m, n = size(M);
    assert(m == n)      
    L = zeros(m, m);

#    degree = zeros(m);
#    for i = 1:m
#        degree[i] += length(find(M[i,:])) + length(find(M[:,i]));
#    end
#    degree = sortperm(degree);
#    M = M[degree, degree];
 
    for i = 1:m
        if M[i,i] < 1e-12 # replace small pivot
            M[i,i] = 1e128;
        end
        L[i,i] = sqrt(M[i,i]);
        
        for j = i+1:m
            L[j,i] = M[j,i] / L[i,i];
            for k = i+1:j
                M[j,k] -= L[j,i]*L[k,i];
            end
        end
    end
    return L 
end
