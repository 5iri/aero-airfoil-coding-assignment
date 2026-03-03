using LinearAlgebra
using Plots
using Printf
using Xfoil

"""
    naca4_camber(code, x, c)

Return camber-line y and dy/dx for a NACA 4-digit airfoil at x.
"""
function naca4_camber(code::AbstractString, x::Float64, c::Float64)
    @assert length(code) == 4 "Use a 4-digit NACA code like \"2412\"."

    m = parse(Int, code[1]) / 100.0
    p = parse(Int, code[2]) / 10.0

    ξ = x / c
    
    if m == 0.0 || p == 0.0
        return 0.0, 0.0
    elseif ξ < p
        yc = m / p^2 * (2.0 * p * ξ - ξ^2) * c
        dyc_dx = 2.0 * m / p^2 * (p - ξ)
        return yc, dyc_dx
    else
        yc = m / (1.0 - p)^2 * ((1.0 - 2.0 * p) + 2.0 * p * ξ - ξ^2) * c
        dyc_dx = 2.0 * m / (1.0 - p)^2 * (p - ξ)
        return yc, dyc_dx
    end
end

"""
    naca4_surface_closed_te(code; n_half=140, c=1.0)

Return a closed-trailing-edge NACA 4-digit boundary for use in XFOIL.
Point order is TE (upper) -> LE -> TE (lower).
"""
function naca4_surface_closed_te(code::AbstractString; n_half::Int=140, c::Float64=1.0)
    @assert length(code) == 4 "Use a 4-digit NACA code like \"2412\"."

    m = parse(Int, code[1]) / 100.0
    p = parse(Int, code[2]) / 10.0
    t = parse(Int, code[3:4]) / 100.0

    β = collect(range(0.0, π, length=n_half + 1))
    x = 0.5 .* c .* (1 .- cos.(β))
    xbar = x ./ c

    # Closed TE coefficient.
    yt = 5.0 * t * c .* (
        0.2969 .* sqrt.(xbar) .- 0.1260 .* xbar .- 0.3516 .* xbar .^ 2 .+
        0.2843 .* xbar .^ 3 .- 0.1036 .* xbar .^ 4
    )

    yc = zeros(length(x))
    dyc_dx = zeros(length(x))
    if m > 0.0 && p > 0.0
        for i in eachindex(xbar)
            ξ = xbar[i]
            if ξ < p
                yc[i] = m / p^2 * (2.0 * p * ξ - ξ^2) * c
                dyc_dx[i] = 2.0 * m / p^2 * (p - ξ)
            else
                yc[i] = m / (1.0 - p)^2 * ((1.0 - 2.0 * p) + 2.0 * p * ξ - ξ^2) * c
                dyc_dx[i] = 2.0 * m / (1.0 - p)^2 * (p - ξ)
            end
        end
    end

    θ = atan.(dyc_dx)
    xu = x .- yt .* sin.(θ)
    yu = yc .+ yt .* cos.(θ)
    xl = x .+ yt .* sin.(θ)
    yl = yc .- yt .* cos.(θ)

    xb = vcat(reverse(xu), xl[2:end])
    yb = vcat(reverse(yu), yl[2:end])

    return xb, yb
end

@inline function point_vortex_velocity_coeff(x::Float64, y::Float64, xv::Float64, yv::Float64)
    dx = x - xv
    dy = y - yv
    r2 = dx * dx + dy * dy + 1e-14
    u = -(1.0 / (2.0 * π)) * (dy / r2)
    v = (1.0 / (2.0 * π)) * (dx / r2)
    return u, v
end

"""
    build_case(code, h_over_c; c=1.0, n_panels=120, include_wall=true)

Build one camber-line collocation case for a given h/c.
"""
function build_case(
    code::AbstractString,
    h_over_c::Float64;
    c::Float64=1.0,
    n_panels::Int=120,
    include_wall::Bool=true,
)
    # Cosine spacing for better leading-edge resolution.
    β = collect(range(0.0, π, length=n_panels + 1))
    x_edge = 0.5 .* c .* (1 .- cos.(β))

    x_ctrl = zeros(n_panels)
    y_ctrl = zeros(n_panels)
    x_vort = zeros(n_panels)
    y_vort = zeros(n_panels)
    tx = zeros(n_panels)
    ty = zeros(n_panels)
    nx = zeros(n_panels)
    ny = zeros(n_panels)
    ds = zeros(n_panels)

    for j in 1:n_panels
        xa = x_edge[j]
        xb = x_edge[j + 1]

        xq = xa + 0.25 * (xb - xa)
        xc = xa + 0.75 * (xb - xa)

        yq0, _ = naca4_camber(code, xq, c)
        yc0, dyc_dx = naca4_camber(code, xc, c)
        ya0, _ = naca4_camber(code, xa, c)
        yb0, _ = naca4_camber(code, xb, c)

        yq = yq0 + h_over_c * c
        yc = yc0 + h_over_c * c

        s = hypot(xb - xa, yb0 - ya0)
        tnorm = hypot(1.0, dyc_dx)

        x_vort[j] = xq
        y_vort[j] = yq
        x_ctrl[j] = xc
        y_ctrl[j] = yc
        ds[j] = s

        tx[j] = 1.0 / tnorm
        ty[j] = dyc_dx / tnorm
        nx[j] = -dyc_dx / tnorm
        ny[j] = 1.0 / tnorm
    end

    bu = zeros(n_panels, n_panels)
    bv = zeros(n_panels, n_panels)
    a = zeros(n_panels, n_panels)

    for i in 1:n_panels
        xi = x_ctrl[i]
        yi = y_ctrl[i]
        for j in 1:n_panels
            u, v = point_vortex_velocity_coeff(xi, yi, x_vort[j], y_vort[j])
            if include_wall
                ui, vi = point_vortex_velocity_coeff(xi, yi, x_vort[j], -y_vort[j])
                # Image vortex has opposite strength for slip wall.
                u -= ui
                v -= vi
            end
            bu[i, j] = u
            bv[i, j] = v
            a[i, j] = nx[i] * u + ny[i] * v
        end
    end

    fac = lu(a)
    perm = sortperm(x_ctrl)

    return (
        fac=fac,
        x_ctrl=x_ctrl,
        y_ctrl=y_ctrl,
        x_vort=x_vort,
        y_vort=y_vort,
        tx=tx,
        ty=ty,
        nx=nx,
        ny=ny,
        ds=ds,
        bu=bu,
        bv=bv,
        perm=perm,
        include_wall=include_wall,
    )
end

function solve_gamma(case_data, alpha_deg::Float64; Uinf::Float64=1.0)
    α = deg2rad(alpha_deg)
    ux = Uinf * cos(α)
    uy = Uinf * sin(α)
    rhs = -(case_data.nx .* ux .+ case_data.ny .* uy)
    return case_data.fac \ rhs
end

raw_cl(Γ, ds; Uinf::Float64=1.0, c::Float64=1.0) = 2.0 * sum(Γ) / (Uinf * c)

function cp_upper_from_gamma(
    case_data,
    Γ,
    alpha_deg::Float64;
    Uinf::Float64=1.0,
    c::Float64=1.0,
    x_clip::Tuple{Float64, Float64}=(0.01, 0.99),
)
    α = deg2rad(alpha_deg)
    ux = Uinf * cos(α)
    uy = Uinf * sin(α)

    u = ux .+ case_data.bu * Γ
    v = uy .+ case_data.bv * Γ

    vt_line = case_data.tx .* u .+ case_data.ty .* v
    gamma_sheet = Γ ./ case_data.ds
    vt_upper = vt_line .- 0.5 .* gamma_sheet
    cp = 1.0 .- (vt_upper ./ Uinf) .^ 2

    x_over_c = case_data.x_ctrl ./ c
    idx_sorted = case_data.perm
    x_sorted = x_over_c[idx_sorted]
    cp_sorted = cp[idx_sorted]

    lo, hi = x_clip
    keep = (x_sorted .>= lo) .& (x_sorted .<= hi)
    return x_sorted[keep], cp_sorted[keep]
end

function gamma_sheet_distribution(
    case_data,
    Γ;
    Uinf::Float64=1.0,
    c::Float64=1.0,
    sign_corr::Float64=1.0,
    x_clip::Tuple{Float64, Float64}=(0.01, 0.99),
)
    γ = sign_corr .* (Γ ./ case_data.ds) ./ Uinf
    x_over_c = case_data.x_ctrl ./ c
    idx_sorted = case_data.perm
    x_sorted = x_over_c[idx_sorted]
    γ_sorted = γ[idx_sorted]

    lo, hi = x_clip
    keep = (x_sorted .>= lo) .& (x_sorted .<= hi)
    return x_sorted[keep], γ_sorted[keep]
end

@inline function linear_interp(x::Vector{Float64}, y::Vector{Float64}, xq::Float64)
    if xq <= x[1]
        return y[1]
    elseif xq >= x[end]
        return y[end]
    end

    i = searchsortedlast(x, xq)
    i = clamp(i, 1, length(x) - 1)
    x1 = x[i]
    x2 = x[i + 1]
    t = (xq - x1) / (x2 - x1)
    return (1.0 - t) * y[i] + t * y[i + 1]
end

function rmse_on_overlap(xa::Vector{Float64}, ya::Vector{Float64}, xb::Vector{Float64}, yb::Vector{Float64})
    lo = max(xa[1], xb[1])
    hi = min(xa[end], xb[end])
    xq = [x for x in xa if (x >= lo && x <= hi)]
    if isempty(xq)
        return NaN
    end

    err2 = 0.0
    for x in xq
        ref = linear_interp(xb, yb, x)
        d = linear_interp(xa, ya, x) - ref
        err2 += d * d
    end
    return sqrt(err2 / length(xq))
end

function xfoil_cp_upper(
    airfoil_code::AbstractString,
    alpha_deg::Float64;
    n_half::Int=120,
    x_clip::Tuple{Float64, Float64}=(0.01, 0.99),
)
    xb, yb = naca4_surface_closed_te(airfoil_code; n_half=n_half, c=1.0)
    Xfoil.set_coordinates(xb, yb)
    Xfoil.pane()
    Xfoil.solve_alpha(alpha_deg)

    xsurf, cpsurf = Xfoil.cpdump()
    ile = argmin(xsurf)
    x_up = xsurf[1:ile]
    cp_up = cpsurf[1:ile]

    perm = sortperm(x_up)
    x_sorted = x_up[perm]
    cp_sorted = cp_up[perm]

    lo, hi = x_clip
    keep = (x_sorted .>= lo) .& (x_sorted .<= hi)
    return x_sorted[keep], cp_sorted[keep]
end

"""
Verification plot:
compare no-ground model Cp with XFOIL Cp for a few AoA.
"""
function make_cp_verification_plot(
    airfoil_code::AbstractString,
    case_no_wall,
    verification_alphas,
    out_path;
    Uinf::Float64=1.0,
    c::Float64=1.0,
    airfoil_name::AbstractString="NACA 2412",
)
    n = length(verification_alphas)
    p = plot(layout=(1, n), size=(1550, 470))

    rmse_lines = String[]
    for (i, α) in enumerate(verification_alphas)
        x_model, cp_model = cp_upper_from_gamma(case_no_wall, solve_gamma(case_no_wall, α; Uinf=Uinf), α; Uinf=Uinf, c=c)
        x_ref, cp_ref = xfoil_cp_upper(airfoil_code, α)

        e = rmse_on_overlap(x_model, cp_model, x_ref, cp_ref)
        push!(rmse_lines, @sprintf("alpha=%0.0f deg: RMSE=%.3f", α, e))

        plot!(p[i], x_model, cp_model, lw=3, label="Model")
        plot!(p[i], x_ref, cp_ref, lw=2, ls=:dash, c=:black, label="XFOIL (Drela)")

        plot!(
            p[i],
            xlabel="x/c",
            ylabel="C_p (upper)",
            title=@sprintf("alpha = %0.0f deg", α),
            yflip=true,
            legend=:topright,
            framestyle=:box,
        )
    end

    plot!(p, plot_title="$(airfoil_name): Cp verification vs XFOIL (no ground)")
    savefig(p, out_path)

    println("Verification summary:")
    for line in rmse_lines
        println("  ", line)
    end

    return p
end

"""
Render geometry/setup diagram for NACA 2412 and all h/c cases.
"""
function make_setup_diagram(
    airfoil_code::AbstractString,
    cases,
    out_path;
    c::Float64=1.0,
    airfoil_name::AbstractString="NACA 2412",
)
    xb, yb = naca4_surface_closed_te(airfoil_code; n_half=160, c=c)

    p = plot(
        xlabel="x/c",
        ylabel="y/c",
        title="$(airfoil_name): geometry and ground-effect setup",
        legend=:outerright,
        framestyle=:box,
        aspect_ratio=:equal,
        size=(1200, 560),
    )

    # Ground wall
    plot!(p, [0.0, 1.0], [0.0, 0.0], c=:black, lw=3, label="ground (y = 0)")

    for case in cases
        if case.include_wall
            ys = yb ./ c .+ case.h_over_c
            plot!(p, xb ./ c, ys, lw=3, label=case.label)
        end
    end

    # No-ground reference position shown separately to avoid overlap in the near-ground region.
    y_offset_nowall = 1.25
    plot!(p, xb ./ c, yb ./ c .+ y_offset_nowall, lw=3, ls=:dash, label="No wall (h/c -> inf)")

    annotate!(p, 0.80, 0.06, text("wall", 10, :left))
    annotate!(p, 0.52, 0.32, text("U∞, alpha", 10, :left))

    # Simple freestream arrow.
    quiver!(
        p,
        [0.30],
        [0.25],
        quiver=([0.20], [0.07]),
        lw=2,
        c=:black,
        label="",
    )

    ylims!(p, -0.06, 1.45)
    xlims!(p, -0.02, 1.05)
    savefig(p, out_path)
    return p
end

function make_cl_plot(cl_alphas, case_data_list, labels, sign_corr, out_path; Uinf=1.0, c=1.0, airfoil_name="NACA 2412")
    p = plot(
        xlabel="alpha (deg)",
        ylabel="C_L",
        title="$(airfoil_name): C_L vs alpha for various h/c",
        lw=3,
        marker=:circle,
        ms=3,
        legend=:topleft,
    )

    for (k, case_data) in enumerate(case_data_list)
        cl_vals = Float64[]
        for α in cl_alphas
            Γ = solve_gamma(case_data, α; Uinf=Uinf)
            cl = sign_corr * raw_cl(Γ, case_data.ds; Uinf=Uinf, c=c)
            push!(cl_vals, cl)
        end
        plot!(p, cl_alphas, cl_vals, label=labels[k])
    end

    hline!(p, [0.0], c=:black, ls=:dash, lw=1, label="")
    savefig(p, out_path)
    return p
end

"""
Required layout:
each subplot is one h/c case, each curve is a different alpha.
"""
function make_cp_plot_grouped_by_height(cp_alphas, case_data_list, labels, out_path; Uinf=1.0, c=1.0, airfoil_name="NACA 2412")
    n = length(case_data_list)
    ncols = 2
    nrows = cld(n, ncols)
    p = plot(layout=(nrows, ncols), size=(1250, 850))

    for (i, case_data) in enumerate(case_data_list)
        for α in cp_alphas
            Γ = solve_gamma(case_data, α; Uinf=Uinf)
            xcp, cp = cp_upper_from_gamma(case_data, Γ, α; Uinf=Uinf, c=c)
            plot!(p[i], xcp, cp, lw=3, label="alpha = $(Int(round(α))) deg")
        end

        plot!(
            p[i],
            xlabel="x/c",
            ylabel="C_p (upper)",
            title=labels[i],
            yflip=true,
            legend=:topright,
            framestyle=:box,
        )
    end

    plot!(p, plot_title="$(airfoil_name): C_p vs x/c (alphas in each h/c panel)")
    savefig(p, out_path)
    return p
end

function make_gamma_plot_grouped_by_height(cp_alphas, case_data_list, labels, sign_corr, out_path; Uinf=1.0, c=1.0, airfoil_name="NACA 2412")
    n = length(case_data_list)
    ncols = 2
    nrows = cld(n, ncols)
    p = plot(layout=(nrows, ncols), size=(1250, 850))

    for (i, case_data) in enumerate(case_data_list)
        for α in cp_alphas
            Γ = solve_gamma(case_data, α; Uinf=Uinf)
            xg, γ = gamma_sheet_distribution(case_data, Γ; Uinf=Uinf, c=c, sign_corr=sign_corr)
            plot!(p[i], xg, γ, lw=3, label="alpha = $(Int(round(α))) deg")
        end

        plot!(
            p[i],
            xlabel="x/c",
            ylabel="gamma/U_inf",
            title=labels[i],
            legend=:topright,
            framestyle=:box,
        )
    end

    plot!(p, plot_title="$(airfoil_name): gamma vs x/c (alphas in each h/c panel)")
    savefig(p, out_path)
    return p
end

function main()
    airfoil_code = "2412"
    airfoil_name = "NACA $airfoil_code"

    c = 1.0
    Uinf = 1.0
    n_panels = 120

    cl_alphas = collect(-8.0:1.0:12.0)
    cp_alphas = [-4.0, 0.0, 4.0, 8.0, 12.0]
    verification_alphas = [0.0, 4.0, 8.0]

    cases = [
        (label="No wall (h/c -> inf)", h_over_c=1.0, include_wall=false),
        (label="h/c = 1.00", h_over_c=1.00, include_wall=true),
        (label="h/c = 0.50", h_over_c=0.50, include_wall=true),
        (label="h/c = 0.15", h_over_c=0.15, include_wall=true),
    ]

    case_data_list = [
        build_case(
            airfoil_code,
            case.h_over_c;
            c=c,
            n_panels=n_panels,
            include_wall=case.include_wall,
        )
        for case in cases
    ]
    labels = [case.label for case in cases]

    # Orient sign so C_L increases with alpha.
    Γp = solve_gamma(case_data_list[1], 4.0; Uinf=Uinf)
    Γm = solve_gamma(case_data_list[1], -4.0; Uinf=Uinf)
    slope_raw = raw_cl(Γp, case_data_list[1].ds; Uinf=Uinf, c=c) -
                raw_cl(Γm, case_data_list[1].ds; Uinf=Uinf, c=c)
    sign_corr = slope_raw >= 0.0 ? 1.0 : -1.0

    out_cl = joinpath(@__DIR__, "naca2412_cl_vs_alpha_heights.png")
    out_cp = joinpath(@__DIR__, "naca2412_cp_vs_xc_by_height_all_alphas.png")
    out_gamma = joinpath(@__DIR__, "naca2412_gamma_vs_xc_by_height_all_alphas.png")
    out_verify = joinpath(@__DIR__, "naca2412_cp_verification_xfoil.png")
    out_setup = joinpath(@__DIR__, "naca2412_setup_diagram.png")

    make_setup_diagram(
        airfoil_code, cases, out_setup;
        c=c, airfoil_name=airfoil_name
    )

    make_cl_plot(
        cl_alphas, case_data_list, labels, sign_corr, out_cl;
        Uinf=Uinf, c=c, airfoil_name=airfoil_name
    )
    make_cp_plot_grouped_by_height(
        cp_alphas, case_data_list, labels, out_cp;
        Uinf=Uinf, c=c, airfoil_name=airfoil_name
    )
    make_gamma_plot_grouped_by_height(
        cp_alphas, case_data_list, labels, sign_corr, out_gamma;
        Uinf=Uinf, c=c, airfoil_name=airfoil_name
    )
    make_cp_verification_plot(
        airfoil_code, case_data_list[1], verification_alphas, out_verify;
        Uinf=Uinf, c=c, airfoil_name=airfoil_name
    )

    println("Airfoil: ", airfoil_name)
    println("Saved plot to: ", out_cl)
    println("Saved plot to: ", out_cp)
    println("Saved plot to: ", out_gamma)
    println("Saved plot to: ", out_verify)
    println("Saved plot to: ", out_setup)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
