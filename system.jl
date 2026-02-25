using LinearAlgebra
using Plots

# --------------------------------------------------
# Build flat-plate vortex-collocation system
# --------------------------------------------------
"""
    panel_geometry(N; c=1.0)

Returns panel edges and collocation geometry for a flat plate on [0, c].
"""
function panel_geometry(N; c=1.0)
    x_edge = collect(range(0.0, c, length=N+1))
    Δx = diff(x_edge)

    x_vort = similar(Δx)
    x_ctrl = similar(Δx)

    for j in 1:N
        xa = x_edge[j]
        xb = x_edge[j+1]
        x_vort[j] = xa + 0.25 * (xb - xa)
        x_ctrl[j] = xa + 0.75 * (xb - xa)
    end

    return x_edge, Δx, x_vort, x_ctrl
end

"""
    solve_gamma_flatplate_wall(N; c=1.0, Uinf=1.0, alpha_deg=5.0, h=1.0, include_wall=true)

Returns:
- x_ctrl : control-point x locations
- gamma  : sheet strength gamma_j ≈ Γ_j / Δx
- Γ      : panel vortex strengths
"""
function solve_gamma_flatplate_wall(N; c=1.0, Uinf=1.0, alpha_deg=5.0, h=1.0, include_wall=true)
    α = deg2rad(alpha_deg)

    _, Δx, x_vort, x_ctrl = panel_geometry(N; c=c)

    # Plate is at y = h (wall at y = 0)
    y_plate = h

    A = zeros(N, N)
    b = fill(-Uinf * sin(α), N)   # freestream normal component moved to RHS

    # Influence coefficients for point vortices
    for i in 1:N
        xi = x_ctrl[i]
        yi = y_plate

        for j in 1:N
            xj = x_vort[j]
            yj = y_plate

            # Real vortex contribution to vertical velocity (normal velocity for flat plate)
            dxr = xi - xj
            dyr = yi - yj  # = 0 here, but keep general
            r2r = dxr^2 + dyr^2

            a_real = (1 / (2π)) * (dxr / r2r)

            a_total = a_real

            if include_wall
                # Image vortex at (xj, -yj) with opposite strength
                dxi = xi - xj
                dyi = yi - (-yj)  # = 2h
                r2i = dxi^2 + dyi^2

                a_img = -(1 / (2π)) * (dxi / r2i)
                a_total += a_img
            end

            A[i, j] = a_total
        end
    end

    # Solve for panel vortex strengths Γ_j
    Γ = A \ b

    # Convert to sheet strength gamma_j ≈ Γ_j / Δx_j
    gamma = Γ ./ Δx

    return x_ctrl, gamma, Γ
end

# --------------------------------------------------
# Post-processing helpers
# --------------------------------------------------
lift_coefficient(Γ; Uinf=1.0, c=1.0) = -2.0 * sum(Γ) / (Uinf * c)

"""
    cp_upper_distribution(N; c=1.0, Uinf=1.0, alpha_deg=0.0, h=1.0, include_wall=true)

Returns the upper-surface pressure coefficient distribution along x/c.
The local tangential speed includes:
- freestream tangential component,
- image-vortex tangential induction (ground effect),
- sheet jump term (-gamma/2) for the upper side.
"""
function cp_upper_distribution(N; c=1.0, Uinf=1.0, alpha_deg=0.0, h=1.0, include_wall=true)
    x_ctrl, gamma, Γ = solve_gamma_flatplate_wall(
        N; c=c, Uinf=Uinf, alpha_deg=alpha_deg, h=h, include_wall=include_wall
    )
    _, _, x_vort, _ = panel_geometry(N; c=c)

    u_image = zeros(N)

    if include_wall
        dy = 2.0 * h
        for i in 1:N
            xi = x_ctrl[i]
            ui = 0.0
            for j in 1:N
                dx = xi - x_vort[j]
                r2 = dx^2 + dy^2
                Γ_img = -Γ[j]
                ui += -(Γ_img / (2π)) * (dy / r2)
            end
            u_image[i] = ui
        end
    end

    U_t = Uinf * cosd(alpha_deg)
    V_upper = U_t .+ u_image .- 0.5 .* gamma
    Cp_upper = 1.0 .- (V_upper ./ Uinf) .^ 2

    return x_ctrl ./ c, Cp_upper
end

function make_cl_plot(N, c, Uinf, alphas, height_cases, out_path)
    p = plot(
        xlabel="α (deg)",
        ylabel="C_L",
        title="C_L vs α for various h/c",
        lw=3,
        marker=:circle,
        ms=3,
        legend=:best,
    )

    for case in height_cases
        cl_values = Float64[]
        for α in alphas
            _, _, Γ = solve_gamma_flatplate_wall(
                N; c=c, Uinf=Uinf, alpha_deg=α, h=case.h, include_wall=case.include_wall
            )
            push!(cl_values, lift_coefficient(Γ; Uinf=Uinf, c=c))
        end
        plot!(p, alphas, cl_values, label=case.label)
    end

    hline!(p, [0.0], c=:black, ls=:dash, lw=1, label="")
    savefig(p, out_path)
    return p
end

function make_cp_plot_grouped_by_height(N, c, Uinf, cp_alphas, height_cases, out_path)
    n = length(height_cases)
    ncols = 2
    nrows = cld(n, ncols)
    p = plot(layout=(nrows, ncols), size=(1200, 800), legend=:topright)

    for (idx, case) in enumerate(height_cases)
        for α in cp_alphas
            x_over_c, cp = cp_upper_distribution(
                N; c=c, Uinf=Uinf, alpha_deg=α, h=case.h, include_wall=case.include_wall
            )
            plot!(p[idx], x_over_c, cp, lw=3, label="α = $(Int(round(α)))°")
        end
        plot!(
            p[idx],
            xlabel="x/c",
            ylabel="C_p",
            title=case.label,
            yflip=true,
            framestyle=:box,
        )
    end

    savefig(p, out_path)
    return p
end

function make_cp_plot_grouped_by_alpha(N, c, Uinf, cp_alphas, height_cases, out_path)
    n = length(cp_alphas)
    ncols = 3
    nrows = cld(n, ncols)
    p = plot(layout=(nrows, ncols), size=(1400, 800), legend=:topright)

    for (idx, α) in enumerate(cp_alphas)
        for case in height_cases
            x_over_c, cp = cp_upper_distribution(
                N; c=c, Uinf=Uinf, alpha_deg=α, h=case.h, include_wall=case.include_wall
            )
            plot!(p[idx], x_over_c, cp, lw=3, label=case.label)
        end
        plot!(
            p[idx],
            xlabel="x/c",
            ylabel="C_p",
            title="α = $(Int(round(α)))°",
            yflip=true,
            framestyle=:box,
        )
    end

    savefig(p, out_path)
    return p
end

function main()
    N = 100
    c = 1.0
    Uinf = 1.0

    cl_alphas = collect(-8.0:1.0:12.0)
    cp_alphas = [-4.0, 0.0, 4.0, 8.0, 12.0]

    height_cases = [
        (label="No wall (h/c → ∞)", h=10.0 * c, include_wall=false),
        (label="h/c = 1.00", h=1.00 * c, include_wall=true),
        (label="h/c = 0.50", h=0.50 * c, include_wall=true),
        (label="h/c = 0.15", h=0.15 * c, include_wall=true),
    ]

    out_cl = joinpath(@__DIR__, "cl_vs_alpha_heights.png")
    out_cp_req = joinpath(@__DIR__, "cp_vs_xc_all_hc_by_alpha.png")
    out_cp_extra = joinpath(@__DIR__, "cp_vs_xc_grouped_by_height.png")

    make_cl_plot(N, c, Uinf, cl_alphas, height_cases, out_cl)
    # Required layout: each panel is one AoA, with all h/c curves together.
    make_cp_plot_grouped_by_alpha(N, c, Uinf, cp_alphas, height_cases, out_cp_req)
    # Optional companion view (opposite grouping).
    make_cp_plot_grouped_by_height(N, c, Uinf, cp_alphas, height_cases, out_cp_extra)

    println("Saved plot to: ", out_cl)
    println("Saved plot to: ", out_cp_req)
    println("Saved plot to: ", out_cp_extra)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
