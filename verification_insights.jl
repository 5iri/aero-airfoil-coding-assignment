using Printf
using Statistics
using Plots
using Xfoil

include(joinpath(@__DIR__, "airfoil_ground_effect.jl"))

@inline rmse(a::AbstractVector, b::AbstractVector) = sqrt(mean((a .- b) .^ 2))

function dedup_sorted_xy(x::Vector{Float64}, y::Vector{Float64})
    perm = sortperm(x)
    xs = x[perm]
    ys = y[perm]

    xo = Float64[]
    yo = Float64[]
    for (xi, yi) in zip(xs, ys)
        if isempty(xo) || abs(xi - xo[end]) > 1e-9
            push!(xo, xi)
            push!(yo, yi)
        else
            # Keep last value at duplicate abscissa.
            yo[end] = yi
        end
    end
    return xo, yo
end

function xfoil_upper_lower_cl(
    airfoil_code::AbstractString,
    alpha_deg::Float64;
    n_half::Int=120,
    x_clip::Tuple{Float64, Float64}=(0.01, 0.99),
)
    xb, yb = naca4_surface_closed_te(airfoil_code; n_half=n_half, c=1.0)
    Xfoil.set_coordinates(xb, yb)
    Xfoil.pane()
    out = Xfoil.solve_alpha(alpha_deg)
    cl = out[1]

    xsurf, cpsurf = Xfoil.cpdump()
    ile = argmin(xsurf)

    x_up_raw = xsurf[1:ile]
    cp_up_raw = cpsurf[1:ile]
    x_lo_raw = xsurf[ile:end]
    cp_lo_raw = cpsurf[ile:end]

    x_up, cp_up = dedup_sorted_xy(x_up_raw, cp_up_raw)
    x_lo, cp_lo = dedup_sorted_xy(x_lo_raw, cp_lo_raw)

    lo, hi = x_clip
    kup = (x_up .>= lo) .& (x_up .<= hi)
    klo = (x_lo .>= lo) .& (x_lo .<= hi)

    return x_up[kup], cp_up[kup], x_lo[klo], cp_lo[klo], cl
end

function gamma_ref_from_xfoil_on_grid(
    x_grid::Vector{Float64},
    x_up::Vector{Float64},
    cp_up::Vector{Float64},
    x_lo::Vector{Float64},
    cp_lo::Vector{Float64},
)
    cp_u = [linear_interp(x_up, cp_up, x) for x in x_grid]
    cp_l = [linear_interp(x_lo, cp_lo, x) for x in x_grid]

    vu = sqrt.(max.(0.0, 1.0 .- cp_u))
    vl = sqrt.(max.(0.0, 1.0 .- cp_l))

    # Approximate equivalent camber-line sheet strength from upper/lower speed split.
    return vl .- vu
end

function main()
    airfoil_code = "2412"
    c = 1.0
    Uinf = 1.0

    verify_alphas = [0.0, 4.0, 8.0]
    cl_alphas = collect(-8.0:1.0:12.0)
    panel_counts = [40, 60, 80, 100, 120, 160, 200]

    # Precompute XFOIL references.
    xfoil_ref = Dict{Float64, NamedTuple}()
    for α in union(verify_alphas, cl_alphas)
        xu, cpu, xl, cpl, cl = xfoil_upper_lower_cl(airfoil_code, α)
        xfoil_ref[α] = (xu=xu, cpu=cpu, xl=xl, cpl=cpl, cl=cl)
    end

    # Baseline case for local Cp-error distribution analysis (N=120, no wall).
    base_case = build_case(airfoil_code, 1.0; c=c, n_panels=120, include_wall=false)
    Γp = solve_gamma(base_case, 4.0; Uinf=Uinf)
    Γm = solve_gamma(base_case, -4.0; Uinf=Uinf)
    slope_raw = raw_cl(Γp, base_case.ds; Uinf=Uinf, c=c) - raw_cl(Γm, base_case.ds; Uinf=Uinf, c=c)
    sign_corr = slope_raw >= 0.0 ? 1.0 : -1.0

    x_grid, _ = cp_upper_from_gamma(base_case, solve_gamma(base_case, 0.0; Uinf=Uinf), 0.0; Uinf=Uinf, c=c)
    abs_err_accum = zeros(length(x_grid))

    local_lines = String[]
    for α in verify_alphas
        Γ = solve_gamma(base_case, α; Uinf=Uinf)
        x_model, cp_model = cp_upper_from_gamma(base_case, Γ, α; Uinf=Uinf, c=c)
        ref = xfoil_ref[α]
        cp_ref = [linear_interp(ref.xu, ref.cpu, x) for x in x_model]
        err = cp_model .- cp_ref
        abs_err = abs.(err)
        abs_err_accum .+= abs_err

        imax = argmax(abs_err)
        imin = argmin(abs_err)
        push!(local_lines, @sprintf(
            "alpha=%0.0f deg: max |ΔCp| at x/c=%.3f, min |ΔCp| at x/c=%.3f",
            α, x_model[imax], x_model[imin]
        ))
    end

    mean_abs_err = abs_err_accum ./ length(verify_alphas)
    imax_mean = argmax(mean_abs_err)
    imin_mean = argmin(mean_abs_err)

    p_err = plot(
        x_grid,
        mean_abs_err,
        lw=3,
        xlabel="x/c",
        ylabel="mean |ΔC_p|",
        title="NACA 2412 no-wall: mean Cp error vs x/c (Model vs XFOIL)",
        legend=false,
        framestyle=:box,
    )
    vline!(p_err, [0.25], lw=2, ls=:dash, c=:black)
    vline!(p_err, [0.75], lw=2, ls=:dot, c=:black)
    annotate!(p_err, 0.27, maximum(mean_abs_err) * 0.92, text("c/4", 10))
    annotate!(p_err, 0.77, maximum(mean_abs_err) * 0.85, text("3c/4", 10))
    scatter!(p_err, [x_grid[imax_mean]], [mean_abs_err[imax_mean]], ms=5, label="")
    scatter!(p_err, [x_grid[imin_mean]], [mean_abs_err[imin_mean]], ms=5, label="")
    out_err = joinpath(@__DIR__, "naca2412_cp_error_profile_no_wall.png")
    savefig(p_err, out_err)

    # Panel count trend metrics.
    cp_rmse_vs_n = Float64[]
    cl_rmse_vs_n = Float64[]

    metric_lines = String[]
    for n in panel_counts
        case_n = build_case(airfoil_code, 1.0; c=c, n_panels=n, include_wall=false)
        Γp_n = solve_gamma(case_n, 4.0; Uinf=Uinf)
        Γm_n = solve_gamma(case_n, -4.0; Uinf=Uinf)
        slope_raw_n = raw_cl(Γp_n, case_n.ds; Uinf=Uinf, c=c) - raw_cl(Γm_n, case_n.ds; Uinf=Uinf, c=c)
        sign_corr_n = slope_raw_n >= 0.0 ? 1.0 : -1.0

        cp_errs = Float64[]
        for α in verify_alphas
            Γ = solve_gamma(case_n, α; Uinf=Uinf)

            x_cp, cp_model = cp_upper_from_gamma(case_n, Γ, α; Uinf=Uinf, c=c)
            ref = xfoil_ref[α]
            cp_ref = [linear_interp(ref.xu, ref.cpu, x) for x in x_cp]
            push!(cp_errs, rmse(cp_model, cp_ref))
        end

        cl_model = Float64[]
        cl_ref = Float64[]
        for α in cl_alphas
            Γ = solve_gamma(case_n, α; Uinf=Uinf)
            push!(cl_model, sign_corr_n * raw_cl(Γ, case_n.ds; Uinf=Uinf, c=c))
            push!(cl_ref, xfoil_ref[α].cl)
        end

        cp_rmse = mean(cp_errs)
        cl_rmse = rmse(cl_model, cl_ref)

        push!(cp_rmse_vs_n, cp_rmse)
        push!(cl_rmse_vs_n, cl_rmse)

        push!(
            metric_lines,
            @sprintf("N=%3d: Cp_RMSE=%.4f, CL_RMSE=%.4f", n, cp_rmse, cl_rmse)
        )
    end

    p_trend = plot(
        panel_counts,
        cp_rmse_vs_n,
        lw=3,
        marker=:circle,
        ms=4,
        label="Cp RMSE (mean over alpha=0,4,8)",
        xlabel="number of panels (N)",
        ylabel="RMSE",
        title="No-wall accuracy trend vs panel count (Model vs XFOIL)",
        framestyle=:box,
    )
    plot!(p_trend, panel_counts, cl_rmse_vs_n, lw=3, marker=:utriangle, ms=4, label="CL RMSE (alpha=-8:12)")
    out_trend = joinpath(@__DIR__, "naca2412_accuracy_vs_panels.png")
    savefig(p_trend, out_trend)

    out_txt = joinpath(@__DIR__, "naca2412_verification_metrics.txt")
    open(out_txt, "w") do io
        println(io, "Local Cp-error locations (N=120, no-wall):")
        for line in local_lines
            println(io, line)
        end
        println(io)
        println(io, @sprintf("Mean over alpha=0,4,8: max at x/c=%.3f, min at x/c=%.3f", x_grid[imax_mean], x_grid[imin_mean]))
        println(io)
        println(io, "Panel-count trend metrics:")
        for line in metric_lines
            println(io, line)
        end
    end

    println("Saved plot to: ", out_err)
    println("Saved plot to: ", out_trend)
    println("Saved metrics to: ", out_txt)
    println("Summary:")
    for line in local_lines
        println("  ", line)
    end
    println(@sprintf("  Mean over alpha=0,4,8: max at x/c=%.3f, min at x/c=%.3f", x_grid[imax_mean], x_grid[imin_mean]))
    for line in metric_lines
        println("  ", line)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
