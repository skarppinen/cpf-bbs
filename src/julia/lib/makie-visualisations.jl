include("../../../config.jl");
include(joinpath(LIB_PATH, "SimpleRaster.jl"));
using Printf

function compare_terrain_sim_anim(d_bbcpf, d_cpfbs; #ctcrw_polysize = 0;
                                  filepath)
    #plot_ctcrw = ctcrw_polysize > 0

    # Extract data.
    npar = d_bbcpf.args["npar"];
    blocksize = d_bbcpf.args["blocksize"];
    data = getfield(d_bbcpf, Symbol("data-ctcrw"));
    Δ = data.dt[1]; #d_bbcpf.args["dt"];
    raster = d_bbcpf.mapraster;
    bbcpf_px = d_bbcpf.pmux;
    bbcpf_py = d_bbcpf.pmuy;
    cpfbs_px = d_cpfbs.pmux;
    cpfbs_py = d_cpfbs.pmuy;
    #if plot_ctcrw
    #    mu_smooth = d_bbcpf.mu_smooth;
    #    P_smooth = d_bbcpf.P_smooth;
    #end

    # Time point.
    i = Observable(1);

    # Basemap with changing title.
    maptitle_bbcpf = lift(i) do i
        string("CPF-BBS(N = ", npar, ")",
               ", blocksize = ", blocksize,
                       ", dt = ", Δ,
                       ", t = ", @sprintf "%6.5f" data.time[i]);
    end
    maptitle_cpfbs = lift(i) do i
        string("CPF-BS(N = ", npar, ")",
               ", dt = ", Δ,
               ", t = ", @sprintf "%6.5f" data.time[i]);
    end

    f = Figure(resolution = (1500, 750))
    g = f[1, 1] = GridLayout(); # Insert grid layout.
    axs = [Axis(g[1, i], aspect = 1.0) for i in 1:2];
    linkaxes!(axs[2], axs[1])

    image!(axs[1], [raster.info.bbox.xmin, raster.info.bbox.xmax],
                           [raster.info.bbox.ymin, raster.info.bbox.ymax],
                           transpose(raster.r))
    image!(axs[2], [raster.info.bbox.xmin, raster.info.bbox.xmax],
                           [raster.info.bbox.ymin, raster.info.bbox.ymax],
                           transpose(raster.r))
    bbcpf_particles = lift(i) do i
       Point2.(view(bbcpf_px, i, :), view(bbcpf_py, i, :));
    end
    cpfbs_particles = lift(i) do i
       Point2.(view(cpfbs_px, i, :), view(cpfbs_py, i, :));
    end
    scatter!(axs[1], bbcpf_particles, color = (:darkgreen, 0.2), markersize = 3, strokewidth = 0);
    scatter!(axs[2], cpfbs_particles, color = (:darkgreen, 0.2), markersize = 3, strokewidth = 0);
    on(maptitle_bbcpf) do title
        axs[1].title = title;
    end
    on(maptitle_cpfbs) do title
        axs[2].title = title;
    end

    # (Constant) observations on map.
    xs = collect(skipmissing(data.y[1, :]))
    ys = collect(skipmissing(data.y[2, :]))
    for i in 1:2
        scatter!(axs[i], xs, ys, color = (:blue, 1.0), marker = :xcross,
                         markersize = 9, strokewidth = 0);
    end
    hideydecorations!(axs[2]) # Hide y axis from right pane.
    colgap!(g, 0) # Set space between columns of plot.

    # # Mean of CTCRW that changes with time.
    # if plot_ctcrw
    #     ctcrw_mean = lift(i) do i
    #         Point2(mu_smooth[i][1], mu_smooth[i][3])
    #     end
    #     scatter!(ctcrw_mean, color = :blue, strokewidth = 0)
    #
    #     # Confidence ellipse of CTCRW.
    #     cov_poly = [zeros(2, ctcrw_polysize) for i in 1:size(data.y, 2)];
    #     for i in 1:size(data.y, 2)
    #         mu = mu_smooth[i][[1, 3]];
    #         cov = P_smooth[i][[1, 3], [1, 3]];
    #         confidence_ellipse!(cov_poly[i], mu, cov);
    #     end
    #     ellipse_points = lift(i) do i
    #         Point2.(cov_poly[i][1, :], cov_poly[i][2, :]);
    #     end
    #     lines!(ellipse_points, color = :blue)
    # end

    record(f, filepath, 1:size(data.y, 2); framerate = 60) do t
        i[] = t
    end

end
using Colors

function plot_trajectory_comparison(filepath::AbstractString,
                                    outfilepath::AbstractString, ntraj = 250;
                                    resolution::NTuple{2} = (1000, 500),
                                    markersize = 10,
                                    linewidth = 3)
    save_fig = if outfilepath == ""
        false;
    else
        true;
    end

    raster, proposal_sim, bbcpf_sim, obsx, obsy = jldopen(filepath, "r") do file
        raster = file["mapraster"];
        proposal_sim = file["proposal-sim"];
        bbcpf_sim = map(file["pmux"], file["pmuy"]) do x, y
            (x, y)
        end
        obsx = collect(skipmissing(file["data-ctcrw"].y[1, :]));
        obsy = collect(skipmissing(file["data-ctcrw"].y[2, :]));
        raster, proposal_sim, bbcpf_sim, obsx, obsy
    end
    trajcolor = colorant"rgba(0, 100, 0, 0.2)"; #"(:darkgreen, 0.2);

    f = Figure(resolution = resolution, fontsize = 10, backgroundcolor = :transparent)
    g = f[1, 1] = GridLayout(); # Insert grid layout.
    axs = [Axis(g[1, i], aspect = 1) for i in 1:2]
    linkaxes!(axs[1], axs[2])
    xlims!(axs[2], low = raster.info.bbox.xmin, high = raster.info.bbox.xmax)
    ylims!(axs[2], low = raster.info.bbox.ymin, high = raster.info.bbox.ymax)
    hidedecorations!(axs[1])
    hidedecorations!(axs[2])

    # Plot background map.
    for i in 1:2
        image!(axs[i], [raster.info.bbox.xmin, raster.info.bbox.xmax],
                       [raster.info.bbox.ymin, raster.info.bbox.ymax],
                       transpose(raster.r))
    end
    for i in 1:ntraj
        lines!(axs[1], map(first, proposal_sim[:, i]), map(x -> x[2], proposal_sim[:, i]),
                              color = trajcolor, linestyle = :dash, linewidth = linewidth)
    end
    for i in Int.(collect(1:(10000 / ntraj):10000))
        lines!(axs[2], map(first, bbcpf_sim[:, i]), map(x -> x[2], bbcpf_sim[:, i]),
                              color = trajcolor, linestyle = :dash, linewidth = linewidth)
    end
    for i in 1:2
        scatter!(axs[i], obsx, obsy, color = (:cyan, 1.0), marker = :xcross, markersize = markersize)
    end
    colgap!(g, 0) # Set space between columns of plot.

    save_fig && (save(outfilepath, f, pt_per_unit = 1););
    f;
end

