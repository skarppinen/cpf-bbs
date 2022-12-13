using Roots

function βx_eq_gen(σ, s11, s22)
    let σ = σ, s11 = s11, s22 = s22
        function(βx)
            σ² = σ * σ;
            σ²_div_2s11 = σ² * inv(2.0 * s11);
            term1 = inv(2.0 * βx);
            term2 = inv(2.0 * σ²_div_2s11)
            term3 = -2.0 * inv(βx + σ²_div_2s11);
            σ² * inv((σ²_div_2s11 - βx) ^ 2) * (term1 + term2 + term3) - s22;
        end
    end
end

function solve_βx(σ, s11, s22, limits = (0.0, 100.0))
    f = βx_eq_gen(σ, s11, s22);
    βx_problem = ZeroProblem(f, limits);
    solve(βx_problem, Bisection());
end
