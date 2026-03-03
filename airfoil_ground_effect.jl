using LinearAlgebra
using Plots
using Printf
using Statistics
using Xfoil

# 6-point Gauss-Legendre quadrature on [0,1]
const GL_XI = [
    0.033765242898423975,
    0.16939530676686776,
    0.38069040695840156,
    0.6193095930415985,
    0.8306046932331322,
    0.966234757101576,
]

const GL_W = [
    0.08566224618958517,
    0.1803807865240693,
    0.23395696728634552,
    0.23395696728634552,
    0.1803807865240693,
    0.08566224618958517,
]

"""
    naca4_surface_closed_te(code; n_half=120, c=1.0)

Build closed-trailing-edge NACA 4-digit coordinates:
TE upper -> LE -> TE lower.
"""
function naca4_surface_closed_te(code::AbstractString; n_half::Int=120, c::Float64=1.0)
    @assert length(code) == 4 "Use 4-digit code, e.g. \"2412\"."

    m = parse(Int, code[1]) / 100.0
    p = parse(Int, code[2]) / 10.0
    t = parse(Int, code[3:4]) / 100.0

    beta = collect(range(0.0, pi, length=n_half + 1))
    x = 0.5 .* c .* (1 .- cos.(beta))
    xi = x ./ c

    yt = 5.0 * t * c .* (
        0.2969 .* sqrt.(xi) .- 0.1260 .* xi .- 0.3516 .* xi .^ 2 .+
        0.2843 .* xi .^ 3 .- 0.1036 .* xi .^ 4
    )

    yc = zeros(length(x))
    dyc_dx = zeros(length(x))
    if m > 0.0 && p > 0.0
        for i in eachindex(xi)
            if xi[i] < p
                yc[i] = m / p^2 * (2.0 * p * xi[i] - xi[i]^2) * c
                dyc_dx[i] = 2.0 * m / p^2 * (p - xi[i])
            else
                yc[i] = m / (1.0 - p)^2 * ((1.0 - 2.0 * p) + 2.0 * p * xi[i] - xi[i]^2) * c
                dyc_dx[i] = 2.0 * m / (1.0 - p)^2 * (p - xi[i])
            end
        end
    end

    theta = atan.(dyc_dx)
    xu = x .- yt .* sin.(theta)
    yu = yc .+ yt .* cos.(theta)
    xl = x .+ yt .* sin.(theta)
    yl = yc .- yt .* cos.(theta)

    xb = vcat(reverse(xu), xl[2:end])
    yb = vcat(reverse(yu), yl[2:end])

    return xb, yb
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

@inline rmse(a::AbstractVector, b::AbstractVector) = sqrt(mean((a .- b) .^ 2))

function rmse_on_overlap(
    xa::Vector{Float64},
    ya::Vector{Float64},
    xb::Vector{Float64},
    yb::Vector{Float64},
)
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

function dedup_sorted_xy(x::Vector{Float64}, y::Vector{Float64})
    perm = sortperm(x)
    xs = x[perm]
    ys = y[perm]

    xo = Float64[]
    yo = Float64[]
    for (xi, yi) in zip(xs, ys)
        if isempty(xo) || abs(xi - xo[end]) > 1e-10
            push!(xo, xi)
            push!(yo, yi)
        else
            yo[end] = yi
        end
    end
    return xo, yo
end

function build_panels(xb::Vector{Float64}, yb::Vector{Float64})
    x = copy(xb)
    y = copy(yb)

    if hypot(x[1] - x[end], y[1] - y[end]) > 1e-12
        push!(x, x[1])
        push!(y, y[1])
    end

    n = length(x) - 1
    @assert iseven(n) "Expected an even number of surface panels."

    x1 = x[1:n]
    y1 = y[1:n]
    x2 = x[2:end]
    y2 = y[2:end]

    dx = x2 .- x1
    dy = y2 .- y1
    s = hypot.(dx, dy)
    @assert all(s .> 1e-12) "Zero-length panel found."

    tx = dx ./ s
    ty = dy ./ s
    xc = 0.5 .* (x1 .+ x2)
    yc = 0.5 .* (y1 .+ y2)

    # Start from right-hand normals and orient outward using centroid test.
    nx = copy(ty)
    ny = copy(-tx)
    cx = mean(x[1:end-1])
    cy = mean(y[1:end-1])
    for i in 1:n
        if (xc[i] - cx) * nx[i] + (yc[i] - cy) * ny[i] < 0.0
            nx[i] = -nx[i]
            ny[i] = -ny[i]
        end
    end

    n_half = n ÷ 2
    upper_idx = collect(1:n_half)
    lower_idx = collect(n_half+1:n)

    return (
        n=n,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        s=s,
        tx=tx,
        ty=ty,
        nx=nx,
        ny=ny,
        xc=xc,
        yc=yc,
        upper_idx=upper_idx,
        lower_idx=lower_idx,
    )
end

@inline function induced_velocity_unit_source_panel(
    xp::Float64,
    yp::Float64,
    x1::Float64,
    y1::Float64,
    x2::Float64,
    y2::Float64,
)
    dx = x2 - x1
    dy = y2 - y1
    l = hypot(dx, dy)
    u = 0.0
    v = 0.0

    for k in eachindex(GL_XI)
        xi = GL_XI[k]
        w = GL_W[k]

        xs = x1 + xi * dx
        ys = y1 + xi * dy

        rx = xp - xs
        ry = yp - ys
        r2 = rx * rx + ry * ry + 1e-14

        fac = l * w / (2.0 * pi * r2)
        u += rx * fac
        v += ry * fac
    end

    return u, v
end

@inline function induced_velocity_unit_vortex_panel(
    xp::Float64,
    yp::Float64,
    x1::Float64,
    y1::Float64,
    x2::Float64,
    y2::Float64,
)
    dx = x2 - x1
    dy = y2 - y1
    l = hypot(dx, dy)
    u = 0.0
    v = 0.0

    for k in eachindex(GL_XI)
        xi = GL_XI[k]
        w = GL_W[k]

        xs = x1 + xi * dx
        ys = y1 + xi * dy

        rx = xp - xs
        ry = yp - ys
        r2 = rx * rx + ry * ry + 1e-14

        fac = l * w / (2.0 * pi * r2)
        u += -ry * fac
        v += rx * fac
    end

    return u, v
end

function source_vortex_velocity_with_ground(
    xp::Float64,
    yp::Float64,
    panels,
    j::Int;
    include_wall::Bool=true,
)
    us, vs = induced_velocity_unit_source_panel(
        xp,
        yp,
        panels.x1[j],
        panels.y1[j],
        panels.x2[j],
        panels.y2[j],
    )
    uv, vv = induced_velocity_unit_vortex_panel(
        xp,
        yp,
        panels.x1[j],
        panels.y1[j],
        panels.x2[j],
        panels.y2[j],
    )

    if include_wall
        usi, vsi = induced_velocity_unit_source_panel(
            xp,
            yp,
            panels.x1[j],
            -panels.y1[j],
            panels.x2[j],
            -panels.y2[j],
        )
        uvi, vvi = induced_velocity_unit_vortex_panel(
            xp,
            yp,
            panels.x1[j],
            -panels.y1[j],
            panels.x2[j],
            -panels.y2[j],
        )

        # Ground-image rule for slip wall:
        # source image: same sign, vortex image: opposite sign.
        us += usi
        vs += vsi
        uv -= uvi
        vv -= vvi
    end

    return us, vs, uv, vv
end

"""
    build_case_surface(code, h_over_c; ...)

Surface panel method (Hess-Smith style):
unknowns are panel source strengths sigma_j and one global vortex gamma.
"""
function build_case_surface(
    code::AbstractString,
    h_over_c::Float64;
    c::Float64=1.0,
    n_half::Int=120,
    include_wall::Bool=true,
    eps_kutta::Float64=2e-4,
    eps_cp::Float64=3e-4,
)
    xb, yb = naca4_surface_closed_te(code; n_half=n_half, c=c)
    yb_shift = yb .+ h_over_c * c
    panels = build_panels(xb, yb_shift)

    n = panels.n
    a_sigma = zeros(n, n)
    a_gamma = zeros(n)

    for i in 1:n
        nxi = panels.nx[i]
        nyi = panels.ny[i]
        xi = panels.xc[i]
        yi = panels.yc[i]

        for j in 1:n
            if i == j
                # Principal-value self term for source panel.
                a_sigma[i, j] = 0.5

                if include_wall
                    # Add image contribution on diagonal (non-singular).
                    usi, vsi = induced_velocity_unit_source_panel(
                        xi,
                        yi,
                        panels.x1[j],
                        -panels.y1[j],
                        panels.x2[j],
                        -panels.y2[j],
                    )
                    a_sigma[i, j] += nxi * usi + nyi * vsi

                    uvi, vvi = induced_velocity_unit_vortex_panel(
                        xi,
                        yi,
                        panels.x1[j],
                        -panels.y1[j],
                        panels.x2[j],
                        -panels.y2[j],
                    )
                    a_gamma[i] += nxi * (-uvi) + nyi * (-vvi)
                end
            else
                us, vs, uv, vv = source_vortex_velocity_with_ground(
                    xi,
                    yi,
                    panels,
                    j;
                    include_wall=include_wall,
                )
                a_sigma[i, j] = nxi * us + nyi * vs
                a_gamma[i] += nxi * uv + nyi * vv
            end
        end
    end

    # Kutta condition at trailing edge from upper/lower side points.
    i_up = 1
    i_lo = n

    xu = panels.xc[i_up] + eps_kutta * c * panels.nx[i_up]
    yu = panels.yc[i_up] + eps_kutta * c * panels.ny[i_up]
    xl = panels.xc[i_lo] + eps_kutta * c * panels.nx[i_lo]
    yl = panels.yc[i_lo] + eps_kutta * c * panels.ny[i_lo]

    kutta_sigma = zeros(n)
    kutta_gamma = 0.0

    for j in 1:n
        usu, vsu, uvu, vvu = source_vortex_velocity_with_ground(
            xu,
            yu,
            panels,
            j;
            include_wall=include_wall,
        )
        usl, vsl, uvl, vvl = source_vortex_velocity_with_ground(
            xl,
            yl,
            panels,
            j;
            include_wall=include_wall,
        )

        kutta_sigma[j] =
            panels.tx[i_up] * usu + panels.ty[i_up] * vsu +
            panels.tx[i_lo] * usl + panels.ty[i_lo] * vsl

        kutta_gamma +=
            panels.tx[i_up] * uvu + panels.ty[i_up] * vvu +
            panels.tx[i_lo] * uvl + panels.ty[i_lo] * vvl
    end

    m = zeros(n + 1, n + 1)
    m[1:n, 1:n] .= a_sigma
    m[1:n, end] .= a_gamma
    m[end, 1:n] .= kutta_sigma
    m[end, end] = kutta_gamma

    return (
        panels=panels,
        fac=lu(m),
        include_wall=include_wall,
        c=c,
        eps_cp=eps_cp,
        i_up=i_up,
        i_lo=i_lo,
        h_over_c=h_over_c,
        ds=panels.s,
        x_ctrl=panels.xc,
        y_ctrl=panels.yc,
        x_vort=panels.xc,
        y_vort=panels.yc,
        tx=panels.tx,
        ty=panels.ty,
        nx=panels.nx,
        ny=panels.ny,
        perm=sortperm(panels.xc),
    )
end

function solve_surface_case(case_data, alpha_deg::Float64; Uinf::Float64=1.0)
    n = case_data.panels.n

    alpha = deg2rad(alpha_deg)
    ux = Uinf * cos(alpha)
    uy = Uinf * sin(alpha)

    rhs = zeros(n + 1)
    rhs[1:n] .= -(case_data.panels.nx .* ux .+ case_data.panels.ny .* uy)
    rhs[end] = -(
        ux * (case_data.panels.tx[case_data.i_up] + case_data.panels.tx[case_data.i_lo]) +
        uy * (case_data.panels.ty[case_data.i_up] + case_data.panels.ty[case_data.i_lo])
    )

    q = case_data.fac \ rhs
    sigma = q[1:n]
    gamma = q[end]

    return sigma, gamma
end

function circulation_cl(case_data, gamma::Float64; Uinf::Float64=1.0, c::Float64=1.0)
    gamma_total = gamma * sum(case_data.ds)
    return 2.0 * gamma_total / (Uinf * c)
end

function surface_cp_upper_lower(
    case_data,
    sigma::Vector{Float64},
    gamma::Float64,
    alpha_deg::Float64;
    Uinf::Float64=1.0,
    c::Float64=1.0,
    x_clip::Tuple{Float64, Float64}=(0.01, 0.99),
)
    alpha = deg2rad(alpha_deg)
    ux_inf = Uinf * cos(alpha)
    uy_inf = Uinf * sin(alpha)

    p = case_data.panels
    n = p.n
    cp = zeros(n)

    for i in 1:n
        xi = p.xc[i] + case_data.eps_cp * c * p.nx[i]
        yi = p.yc[i] + case_data.eps_cp * c * p.ny[i]

        u = ux_inf
        v = uy_inf

        for j in 1:n
            us, vs, uv, vv = source_vortex_velocity_with_ground(
                xi,
                yi,
                p,
                j;
                include_wall=case_data.include_wall,
            )
            u += sigma[j] * us + gamma * uv
            v += sigma[j] * vs + gamma * vv
        end

        cp[i] = 1.0 - (u * u + v * v) / (Uinf * Uinf)
    end

    x_over_c = p.xc ./ c

    x_u = x_over_c[p.upper_idx]
    cp_u = cp[p.upper_idx]
    pu = sortperm(x_u)
    x_u = x_u[pu]
    cp_u = cp_u[pu]

    x_l = x_over_c[p.lower_idx]
    cp_l = cp[p.lower_idx]
    pl = sortperm(x_l)
    x_l = x_l[pl]
    cp_l = cp_l[pl]

    lo, hi = x_clip
    keep_u = (x_u .>= lo) .& (x_u .<= hi)
    keep_l = (x_l .>= lo) .& (x_l .<= hi)

    return x_u[keep_u], cp_u[keep_u], x_l[keep_l], cp_l[keep_l]
end

# ---------------- Compatibility wrappers (old names) ----------------

"""
    build_case(code, h_over_c; c=1.0, n_panels=120, include_wall=true)

Wrapper name kept for compatibility. `n_panels` is interpreted as total panel count.
"""
function build_case(
    code::AbstractString,
    h_over_c::Float64;
    c::Float64=1.0,
    n_panels::Int=120,
    include_wall::Bool=true,
)
    n_half = max(20, Int(cld(n_panels, 2)))
    return build_case_surface(
        code,
        h_over_c;
        c=c,
        n_half=n_half,
        include_wall=include_wall,
    )
end

"""
    solve_gamma(case_data, alpha_deg; Uinf=1.0)

Compatibility wrapper: returns a named tuple with `sigma`, `gamma`, `alpha_deg`.
"""
function solve_gamma(case_data, alpha_deg::Float64; Uinf::Float64=1.0)
    sigma, gamma = solve_surface_case(case_data, alpha_deg; Uinf=Uinf)
    return (sigma=sigma, gamma=gamma, alpha_deg=alpha_deg)
end

function raw_cl(sol, ds; Uinf::Float64=1.0, c::Float64=1.0)
    if sol isa NamedTuple && haskey(sol, :gamma)
        return 2.0 * sol.gamma * sum(ds) / (Uinf * c)
    elseif sol isa AbstractVector
        return 2.0 * sum(sol) / (Uinf * c)
    else
        error("raw_cl expects a (sigma,gamma,...) named tuple or a gamma-vector.")
    end
end

function cp_upper_from_gamma(
    case_data,
    sol,
    alpha_deg::Float64;
    Uinf::Float64=1.0,
    c::Float64=1.0,
    x_clip::Tuple{Float64, Float64}=(0.01, 0.99),
)
    if !(sol isa NamedTuple && haskey(sol, :sigma) && haskey(sol, :gamma))
        error("cp_upper_from_gamma expects solve_gamma(...) output.")
    end

    x_u, cp_u, _, _ = surface_cp_upper_lower(
        case_data,
        sol.sigma,
        sol.gamma,
        alpha_deg;
        Uinf=Uinf,
        c=c,
        x_clip=x_clip,
    )

    return x_u, cp_u
end

function gamma_proxy_from_cp(
    x_u::Vector{Float64},
    cp_u::Vector{Float64},
    x_l::Vector{Float64},
    cp_l::Vector{Float64};
    sign_corr::Float64=1.0,
)
    lo = max(x_u[1], x_l[1])
    hi = min(x_u[end], x_l[end])

    xg = [x for x in x_u if (x >= lo && x <= hi)]
    if isempty(xg)
        return Float64[], Float64[]
    end

    vu = sqrt.(max.(0.0, 1.0 .- [linear_interp(x_u, cp_u, x) for x in xg]))
    vl = sqrt.(max.(0.0, 1.0 .- [linear_interp(x_l, cp_l, x) for x in xg]))

    gamma_proxy = sign_corr .* (vl .- vu)
    return xg, gamma_proxy
end

function gamma_sheet_distribution(
    case_data,
    sol;
    Uinf::Float64=1.0,
    c::Float64=1.0,
    sign_corr::Float64=1.0,
    x_clip::Tuple{Float64, Float64}=(0.01, 0.99),
)
    if !(sol isa NamedTuple && haskey(sol, :sigma) && haskey(sol, :gamma) && haskey(sol, :alpha_deg))
        error("gamma_sheet_distribution expects solve_gamma(...) output.")
    end

    x_u, cp_u, x_l, cp_l = surface_cp_upper_lower(
        case_data,
        sol.sigma,
        sol.gamma,
        sol.alpha_deg;
        Uinf=Uinf,
        c=c,
        x_clip=x_clip,
    )

    xg, gg = gamma_proxy_from_cp(x_u, cp_u, x_l, cp_l; sign_corr=sign_corr)
    return xg, gg
end

# ---------------- XFOIL references ----------------

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

function xfoil_cp_upper(
    airfoil_code::AbstractString,
    alpha_deg::Float64;
    n_half::Int=120,
    x_clip::Tuple{Float64, Float64}=(0.01, 0.99),
)
    x_u, cp_u, _, _, _ = xfoil_upper_lower_cl(
        airfoil_code,
        alpha_deg;
        n_half=n_half,
        x_clip=x_clip,
    )
    return x_u, cp_u
end

# ---------------- Plotting ----------------

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

    plot!(p, [0.0, 1.0], [0.0, 0.0], c=:black, lw=3, label="ground (y = 0)")

    for case in cases
        if case.include_wall
            ys = yb ./ c .+ case.h_over_c
            plot!(p, xb ./ c, ys, lw=3, label=case.label)
        end
    end

    y_offset_nowall = 1.25
    plot!(
        p,
        xb ./ c,
        yb ./ c .+ y_offset_nowall,
        lw=3,
        ls=:dash,
        label="No wall (h/c -> inf)",
    )

    annotate!(p, 0.80, 0.06, text("wall", 10, :left))

    quiver!(
        p,
        [0.30],
        [0.25],
        quiver=([0.20], [0.07]),
        lw=2,
        c=:black,
        label="",
    )
    annotate!(p, 0.52, 0.32, text("Uinf, alpha", 10, :left))

    ylims!(p, -0.06, 1.45)
    xlims!(p, -0.02, 1.05)

    savefig(p, out_path)
    return p
end

function make_cl_plot(
    cl_alphas,
    case_data_list,
    labels,
    sign_corr,
    out_path;
    Uinf::Float64=1.0,
    c::Float64=1.0,
    airfoil_name::AbstractString="NACA 2412",
)
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
        for alpha in cl_alphas
            sol = solve_gamma(case_data, alpha; Uinf=Uinf)
            cl = sign_corr * raw_cl(sol, case_data.ds; Uinf=Uinf, c=c)
            push!(cl_vals, cl)
        end
        plot!(p, cl_alphas, cl_vals, label=labels[k])
    end

    hline!(p, [0.0], c=:black, ls=:dash, lw=1, label="")
    savefig(p, out_path)
    return p
end

"""
Requested layout:
one subplot per h/c,
all requested alpha curves in each subplot.
"""
function make_cp_plot_grouped_by_height(
    cp_alphas,
    case_data_list,
    labels,
    out_path;
    Uinf::Float64=1.0,
    c::Float64=1.0,
    airfoil_name::AbstractString="NACA 2412",
)
    n = length(case_data_list)
    ncols = 2
    nrows = cld(n, ncols)
    p = plot(layout=(nrows, ncols), size=(1250, 850))

    for (i, case_data) in enumerate(case_data_list)
        for alpha in cp_alphas
            sol = solve_gamma(case_data, alpha; Uinf=Uinf)
            xcp, cp = cp_upper_from_gamma(case_data, sol, alpha; Uinf=Uinf, c=c)
            plot!(p[i], xcp, cp, lw=3, label="alpha = $(Int(round(alpha))) deg")
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

function make_gamma_plot_grouped_by_height(
    cp_alphas,
    case_data_list,
    labels,
    sign_corr,
    out_path;
    Uinf::Float64=1.0,
    c::Float64=1.0,
    airfoil_name::AbstractString="NACA 2412",
)
    n = length(case_data_list)
    ncols = 2
    nrows = cld(n, ncols)
    p = plot(layout=(nrows, ncols), size=(1250, 850))

    for (i, case_data) in enumerate(case_data_list)
        for alpha in cp_alphas
            sol = solve_gamma(case_data, alpha; Uinf=Uinf)
            xg, gamma_d = gamma_sheet_distribution(
                case_data,
                sol;
                Uinf=Uinf,
                c=c,
                sign_corr=sign_corr,
            )
            plot!(p[i], xg, gamma_d, lw=3, label="alpha = $(Int(round(alpha))) deg")
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

    plot!(p, plot_title="$(airfoil_name): gamma distribution vs x/c (alphas in each h/c panel)")
    savefig(p, out_path)
    return p
end

"""
Verification plot:
compare no-wall model Cp with XFOIL Cp for a few AoA.
"""
function make_cp_verification_plot(
    airfoil_code::AbstractString,
    case_no_wall,
    verification_alphas,
    out_path;
    Uinf::Float64=1.0,
    c::Float64=1.0,
    sign_corr::Float64=1.0,
    airfoil_name::AbstractString="NACA 2412",
)
    n = length(verification_alphas)
    p = plot(layout=(1, n), size=(1550, 470))

    metrics = NamedTuple[]

    for (i, alpha) in enumerate(verification_alphas)
        sol = solve_gamma(case_no_wall, alpha; Uinf=Uinf)
        x_model, cp_model = cp_upper_from_gamma(case_no_wall, sol, alpha; Uinf=Uinf, c=c)

        x_ref, cp_ref, _, _, cl_ref = xfoil_upper_lower_cl(airfoil_code, alpha)

        cp_rmse = rmse_on_overlap(x_model, cp_model, x_ref, cp_ref)
        cl_model = sign_corr * raw_cl(sol, case_no_wall.ds; Uinf=Uinf, c=c)
        delta_cl = cl_model - cl_ref

        push!(
            metrics,
            (
                alpha=alpha,
                cp_rmse=cp_rmse,
                cl_model=cl_model,
                cl_ref=cl_ref,
                delta_cl=delta_cl,
            ),
        )

        plot!(p[i], x_model, cp_model, lw=3, label="Model")
        plot!(p[i], x_ref, cp_ref, lw=2, ls=:dash, c=:black, label="XFOIL (Drela)")

        plot!(
            p[i],
            xlabel="x/c",
            ylabel="C_p (upper)",
            title=@sprintf("alpha = %0.0f deg", alpha),
            yflip=true,
            legend=:topright,
            framestyle=:box,
        )
    end

    plot!(p, plot_title="$(airfoil_name): Cp verification vs XFOIL (no ground)")
    savefig(p, out_path)
    return metrics
end

function make_cp_error_profile_plot(
    airfoil_code::AbstractString,
    case_no_wall,
    verification_alphas,
    out_path;
    Uinf::Float64=1.0,
    c::Float64=1.0,
)
    x_grid = collect(range(0.02, 0.98, length=180))
    mean_abs_err = zeros(length(x_grid))

    for alpha in verification_alphas
        sol = solve_gamma(case_no_wall, alpha; Uinf=Uinf)
        x_model, cp_model = cp_upper_from_gamma(case_no_wall, sol, alpha; Uinf=Uinf, c=c)
        x_ref, cp_ref = xfoil_cp_upper(airfoil_code, alpha)

        for (k, x) in enumerate(x_grid)
            d = linear_interp(x_model, cp_model, x) - linear_interp(x_ref, cp_ref, x)
            mean_abs_err[k] += abs(d)
        end
    end

    mean_abs_err ./= length(verification_alphas)

    imax = argmax(mean_abs_err)
    imin = argmin(mean_abs_err)

    p = plot(
        x_grid,
        mean_abs_err,
        lw=3,
        xlabel="x/c",
        ylabel="mean |Delta C_p|",
        title="NACA 2412 no-wall: mean Cp error vs x/c (Model vs XFOIL)",
        legend=false,
        framestyle=:box,
    )

    vline!(p, [0.25], lw=2, ls=:dash, c=:black)
    vline!(p, [0.75], lw=2, ls=:dot, c=:black)
    annotate!(p, 0.27, maximum(mean_abs_err) * 0.92, text("c/4", 10))
    annotate!(p, 0.77, maximum(mean_abs_err) * 0.85, text("3c/4", 10))

    scatter!(p, [x_grid[imax]], [mean_abs_err[imax]], ms=6, c=:purple, label="")
    scatter!(p, [x_grid[imin]], [mean_abs_err[imin]], ms=6, c=:orange, label="")

    savefig(p, out_path)

    return (
        x_at_max=x_grid[imax],
        x_at_min=x_grid[imin],
        val_at_c4=linear_interp(x_grid, mean_abs_err, 0.25),
        val_at_3c4=linear_interp(x_grid, mean_abs_err, 0.75),
    )
end

function make_accuracy_vs_panels_plot(
    airfoil_code::AbstractString,
    panel_counts,
    verify_alphas,
    cl_alphas,
    out_path;
    Uinf::Float64=1.0,
    c::Float64=1.0,
)
    all_alphas = sort(unique(vcat(verify_alphas, cl_alphas)))
    xfoil_ref = Dict{Float64, NamedTuple}()

    for alpha in all_alphas
        xu, cpu, _, _, cl = xfoil_upper_lower_cl(airfoil_code, alpha)
        xfoil_ref[alpha] = (xu=xu, cpu=cpu, cl=cl)
    end

    cp_rmse_vs_n = Float64[]
    cl_rmse_vs_n = Float64[]
    table_lines = String[]

    for n in panel_counts
        case_n = build_case(airfoil_code, 1.0; c=c, n_panels=n, include_wall=false)

        sol_p = solve_gamma(case_n, 4.0; Uinf=Uinf)
        sol_m = solve_gamma(case_n, -4.0; Uinf=Uinf)
        slope_raw = raw_cl(sol_p, case_n.ds; Uinf=Uinf, c=c) - raw_cl(sol_m, case_n.ds; Uinf=Uinf, c=c)
        sign_corr_n = slope_raw >= 0.0 ? 1.0 : -1.0

        cp_errs = Float64[]
        for alpha in verify_alphas
            sol = solve_gamma(case_n, alpha; Uinf=Uinf)
            x_m, cp_m = cp_upper_from_gamma(case_n, sol, alpha; Uinf=Uinf, c=c)
            ref = xfoil_ref[alpha]
            push!(cp_errs, rmse_on_overlap(x_m, cp_m, ref.xu, ref.cpu))
        end

        cl_model = Float64[]
        cl_ref = Float64[]
        for alpha in cl_alphas
            sol = solve_gamma(case_n, alpha; Uinf=Uinf)
            push!(cl_model, sign_corr_n * raw_cl(sol, case_n.ds; Uinf=Uinf, c=c))
            push!(cl_ref, xfoil_ref[alpha].cl)
        end

        cp_rmse = mean(cp_errs)
        cl_rmse = rmse(cl_model, cl_ref)

        push!(cp_rmse_vs_n, cp_rmse)
        push!(cl_rmse_vs_n, cl_rmse)
        push!(table_lines, @sprintf("N=%3d: Cp_RMSE=%.4f, CL_RMSE=%.4f", n, cp_rmse, cl_rmse))
    end

    p = plot(
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
    plot!(
        p,
        panel_counts,
        cl_rmse_vs_n,
        lw=3,
        marker=:utriangle,
        ms=4,
        label="CL RMSE (alpha=-8:12)",
    )

    savefig(p, out_path)
    return table_lines
end

# ---------------- Main ----------------

function main()
    airfoil_code = "2412"
    airfoil_name = "NACA $airfoil_code"

    c = 1.0
    Uinf = 1.0
    n_panels = 120

    cl_alphas = collect(-8.0:1.0:12.0)
    cp_alphas = [-4.0, 0.0, 4.0, 8.0, 12.0]
    verification_alphas = [0.0, 4.0, 8.0]
    panel_counts = [40, 60, 80, 100, 120, 160, 200]

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
    sol_p = solve_gamma(case_data_list[1], 4.0; Uinf=Uinf)
    sol_m = solve_gamma(case_data_list[1], -4.0; Uinf=Uinf)
    slope_raw = raw_cl(sol_p, case_data_list[1].ds; Uinf=Uinf, c=c) -
                raw_cl(sol_m, case_data_list[1].ds; Uinf=Uinf, c=c)
    sign_corr = slope_raw >= 0.0 ? 1.0 : -1.0

    out_setup = joinpath(@__DIR__, "naca2412_setup_diagram.png")
    out_cl = joinpath(@__DIR__, "naca2412_cl_vs_alpha_heights.png")
    out_cp = joinpath(@__DIR__, "naca2412_cp_vs_xc_by_height_all_alphas.png")
    out_gamma = joinpath(@__DIR__, "naca2412_gamma_vs_xc_by_height_all_alphas.png")
    out_verify = joinpath(@__DIR__, "naca2412_cp_verification_xfoil.png")
    out_err = joinpath(@__DIR__, "naca2412_cp_error_profile_no_wall.png")
    out_trend = joinpath(@__DIR__, "naca2412_accuracy_vs_panels.png")
    out_txt = joinpath(@__DIR__, "naca2412_verification_metrics.txt")

    make_setup_diagram(
        airfoil_code,
        cases,
        out_setup;
        c=c,
        airfoil_name=airfoil_name,
    )

    make_cl_plot(
        cl_alphas,
        case_data_list,
        labels,
        sign_corr,
        out_cl;
        Uinf=Uinf,
        c=c,
        airfoil_name=airfoil_name,
    )

    make_cp_plot_grouped_by_height(
        cp_alphas,
        case_data_list,
        labels,
        out_cp;
        Uinf=Uinf,
        c=c,
        airfoil_name=airfoil_name,
    )

    make_gamma_plot_grouped_by_height(
        cp_alphas,
        case_data_list,
        labels,
        sign_corr,
        out_gamma;
        Uinf=Uinf,
        c=c,
        airfoil_name=airfoil_name,
    )

    verify_metrics = make_cp_verification_plot(
        airfoil_code,
        case_data_list[1],
        verification_alphas,
        out_verify;
        Uinf=Uinf,
        c=c,
        sign_corr=sign_corr,
        airfoil_name=airfoil_name,
    )

    cp_err_stats = make_cp_error_profile_plot(
        airfoil_code,
        case_data_list[1],
        verification_alphas,
        out_err;
        Uinf=Uinf,
        c=c,
    )

    panel_lines = make_accuracy_vs_panels_plot(
        airfoil_code,
        panel_counts,
        verification_alphas,
        cl_alphas,
        out_trend;
        Uinf=Uinf,
        c=c,
    )

    open(out_txt, "w") do io
        println(io, "Verification metrics (Model vs XFOIL)")
        println(io, "")

        for m in verify_metrics
            println(
                io,
                @sprintf(
                    "alpha=%0.0f deg: Cp_RMSE=%.4f, CL_model=%.4f, CL_xfoil=%.4f, DeltaCL=%.4f",
                    m.alpha,
                    m.cp_rmse,
                    m.cl_model,
                    m.cl_ref,
                    m.delta_cl,
                ),
            )
        end

        println(io, "")
        println(io, @sprintf("mean |DeltaCp| max location x/c=%.3f", cp_err_stats.x_at_max))
        println(io, @sprintf("mean |DeltaCp| min location x/c=%.3f", cp_err_stats.x_at_min))
        println(io, @sprintf("mean |DeltaCp|(c/4)=%.4f", cp_err_stats.val_at_c4))
        println(io, @sprintf("mean |DeltaCp|(3c/4)=%.4f", cp_err_stats.val_at_3c4))

        println(io, "")
        println(io, "Panel trend:")
        for line in panel_lines
            println(io, line)
        end
    end

    println("Airfoil: ", airfoil_name)
    println("Saved plot to: ", out_setup)
    println("Saved plot to: ", out_cl)
    println("Saved plot to: ", out_cp)
    println("Saved plot to: ", out_gamma)
    println("Saved plot to: ", out_verify)
    println("Saved plot to: ", out_err)
    println("Saved plot to: ", out_trend)
    println("Saved metrics to: ", out_txt)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
