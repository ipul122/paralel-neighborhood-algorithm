import numpy as NP
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
matplotlib.use("Agg")
from scipy.spatial import Voronoi, voronoi_plot_2d



def rosenbrock(params, a=1, b=100):
    x = params[:, 0]
    y = params[:, 1]
    return NP.log10((a - x) ** 2 + b * (y - x**2) ** 2)


def build_profile(res, thk):
    depth = NP.concatenate(([0], NP.cumsum(thk)))
    res_profile = []
    z_profile = []
    for i in range(len(thk)):
        res_profile.extend([res[i], res[i]])
        z_profile.extend([depth[i], depth[i+1]])
    # half-space
    res_profile.extend([res[-1], res[-1]])
    z_profile.extend([depth[-1], depth[-1] + 10000])  
    return res_profile, z_profile


def plot_all_models(models, a, b, ResReal, ThkReal, n=4, save_path="model.png", depthmax=2000):

    zmax = depthmax
    nz = 50  
    nres = 25
    
    z_grid = NP.linspace(0, zmax, nz)
    res_grid = NP.logspace(0, 4, nres)
    density = NP.zeros((nz-1, nres-1), dtype=NP.float32)
    
    for model in models:
        RES, THK = model[:n], model[n:]
        res_prof, z_prof = build_profile(RES, THK)
        res_interp = NP.interp(z_grid, z_prof, res_prof)
        
        j_indices = NP.searchsorted(res_grid, res_interp[:-1]) - 1
        valid_mask = (j_indices >= 0) & (j_indices < nres-1)
        
        rows = NP.where(valid_mask)[0]
        cols = j_indices[valid_mask]
        density[rows, cols] += 1
    
    max_val = NP.max(density)
    if max_val > 0:
        density /= max_val

    fig, ax = plt.subplots(figsize=(6, 10), dpi=100)
    
    im = ax.pcolormesh(res_grid, z_grid, density, 
                       shading='auto', cmap='viridis')
    
    res_ref, z_ref = build_profile(ResReal, ThkReal)
    ax.step(res_ref, z_ref, where="post", 
            linewidth=3.5, color="black", label="Synthetic")
    
    res_best, z_best = build_profile(a, b)
    ax.step(res_best, z_best, where="post", 
            linewidth=3.5, linestyle="--", color="red", label="Best")
   
    ax.set_xscale("log")
    ax.set_xlim(1, 10000)
    ax.set_ylim(zmax, 0)  
    
    ax.set_xlabel(r"Resistivity ($\Omega\cdot$m)", fontsize=10)
    ax.set_ylabel("Depth (m)", fontsize=10)
    ax.set_title(f"Model (N={len(models)})", fontsize=11)
    
    ax.set_xticks([1, 10, 100, 1000, 10000])
    ax.set_xticklabels([r'$10^0$', r'$10^1$', r'$10^2$', r'$10^3$', r'$10^4$'])  
    
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    plt.colorbar(im, cax=cax, label="Density")
    
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
    
    plt.close(fig)  
    return save_path 

def plot_rosenbrock(models, best_model, bg, bg_misfit, save_path="rosenbrock.png"):
    x_min, x_max, y_min, y_max = -2, 2, -1, 3
    boundary = NP.array([[x_min,y_min],[x_min,y_max],[x_max,y_min],[x_max,y_max]])
    vor = Voronoi(NP.vstack([models, boundary]))

    plt.figure(figsize=(8,6))
    voronoi_plot_2d(vor, show_points=False, show_vertices=False,
                    line_colors='black', line_width=0.5)

    sc = plt.scatter(bg[:,0], bg[:,1], c=bg_misfit, cmap='rainbow', s=50, alpha=0.5, zorder=1)
        
    plt.scatter(models[:,0], models[:,1], s=2, c='black',alpha=0.5, zorder=2, label='Models')
    plt.scatter(1, 1, s=120, c='blue', marker='x', lw=2,
                zorder=10, label='True Optimum')
    plt.scatter(best_model[0], best_model[1], s=120, c='red', marker='+', lw=2,
                zorder=11, label='Best Model')


    plt.xlim(x_min, x_max); plt.ylim(y_min, y_max)
    plt.xlabel('x'); plt.ylabel('y'); plt.title(f'Rosenbrock Function {len(models)} Model')
    plt.colorbar(sc, label='Misfit'); plt.legend()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_obs(
    frequencies,
    Obsres, Obsphs,
    res_best, phs_best,
    EsembleRes, EsemblePhs,
    save_path="curve.png"
):
 

    EsembleRes, EsemblePhs = NP.atleast_2d(EsembleRes), NP.atleast_2d(EsemblePhs)
    res_best = NP.squeeze(res_best)
    phs_best = NP.squeeze(phs_best)
  
    period = 1.0 / frequencies

    nP = len(period)
    nRes = 50
    nPhs = 50

    res_grid = NP.logspace(-1, 4, nRes)
    phs_grid = NP.logspace(-1, 2, nPhs)

    density_res = NP.zeros((nP-1, nRes-1), dtype=NP.float32)
    density_phs = NP.zeros((nP-1, nPhs-1), dtype=NP.float32)

    for i in range(EsembleRes.shape[0]):

        j_res = NP.searchsorted(res_grid, EsembleRes[i, :-1]) - 1
        valid = (j_res >= 0) & (j_res < nRes-1)
        rows = NP.where(valid)[0]
        density_res[rows, j_res[valid]] += 1

        j_phs = NP.searchsorted(phs_grid, EsemblePhs[i, :-1]) - 1
        valid = (j_phs >= 0) & (j_phs < nPhs-1)
        rows = NP.where(valid)[0]
        density_phs[rows, j_phs[valid]] += 1

    if density_res.max() > 0:
        density_res /= density_res.max()
    if density_phs.max() > 0:
        density_phs /= density_phs.max()

    fig = plt.figure(figsize=(8, 10), dpi=120)
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 2])

    ax0 = fig.add_subplot(gs[0]) 
    ax1 = fig.add_subplot(gs[1]) 

    im0 = ax0.pcolormesh(period, res_grid, density_res.T, shading="auto", cmap="viridis")
    ax0.loglog(period, res_best, "r-", linewidth=4, label="Best")
    ax0.loglog(period, Obsres, "ko", markersize=10, label="Observed")
    ax0.set_ylabel("Apparent Resistivity (Ohm.m)")
    ax0.set_title("Apparent Resistivity")
    ax0.set_ylim(1e-0, 1e4)
    ax0.legend(fontsize=8)

    im1 = ax1.pcolormesh(period, phs_grid, density_phs.T, shading="auto", cmap="viridis")
    ax1.semilogx(period, phs_best, "r-", linewidth=4)
    ax1.semilogx(period, Obsphs, "ko", markersize=10)
    ax1.set_xlabel("Period (s)")
    ax1.set_ylabel("Phase (deg)")
    ax1.set_ylim(0, 90)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def create_bounds(nlayer, rho_min=1, rho_max=2000, thk_min=1, thk_max=1100):
  
    lb_res = NP.full(nlayer, rho_min)
    ub_res = NP.full(nlayer, rho_max)
    
    lb_thk = NP.full(nlayer-1, thk_min)
    ub_thk = NP.full(nlayer-1, thk_max)
    
    lb = NP.concatenate([lb_res, lb_thk])
    ub = NP.concatenate([ub_res, ub_thk])
    
    return lb, ub


def corner_plot(
    type,
    samples,
    mean=None,
    best_model=None,
    true_model=None,
    bins=100,
    alpha_scatter=0.2,
):
    n_param = samples.shape[1]

  
    if type == 1:
        param_names = ["x", "y"]
        title = "Rosenbrock Function Parameter"
        bins = 20
    else:
        n_thk = n_param // 2
        n_res = n_param - n_thk
        param_names = (
            [f"RES_{i+1}" for i in range(n_res)] +
            [f"THK_{i+1}" for i in range(n_thk)]
        )
        title = f"MT 1D Inversion ({n_param//2+1} Layer)"

  
    
    fig, axes = plt.subplots(
        n_param, n_param,
        figsize=(2.5 * n_param, 2.5 * n_param),
        squeeze=False
    )

    for i in range(n_param):
        for j in range(n_param):
            ax = axes[i, j]

            if j > i:
                ax.axis("off")
                continue

           
            if i == j:
                orientation = "vertical"
                if j==1 & type == 1:
                    orientation = "horizontal"
                    
                ax.hist(
                    samples[:, i],
                    bins=bins,
                    histtype="step",
                    color="black",
                    orientation=orientation
                )

                if j==1 & type == 1:
                    if true_model is not None:
                        ax.axhline(true_model[i], c="g", lw=3, label="True")

                    if best_model is not None:
                        ax.axhline(best_model[i], c="r", ls="--", lw=2, label="Best")

                    if mean is not None:
                        ax.axhline(mean[i], c="b", ls=":", lw=2, label="Mean")

                    if i == 0:
                        ax.legend(fontsize=8)
                else :
                    if true_model is not None:
                        ax.axvline(true_model[i], c="g", lw=3, label="True")

                    if best_model is not None:
                        ax.axvline(best_model[i], c="r", ls="--", lw=2, label="Best")

                    if mean is not None:
                        ax.axvline(mean[i], c="b", ls=":", lw=2, label="Mean")

                    if i == 0:
                        ax.legend(fontsize=8)

          
            else:
                ax.scatter(
                    samples[:, j],
                    samples[:, i],
                    s=1,
                    alpha=alpha_scatter,
                    color="black"
                )

                if true_model is not None:
                    ax.scatter(
                        true_model[j],
                        true_model[i],
                        c="g",
                        marker="o",
                        s=80,
                        zorder=10
                    )

                if best_model is not None:
                    ax.scatter(
                        best_model[j],
                        best_model[i],
                        c="r",
                        marker="+",
                        s=80,
                        zorder=10
                    )

                if mean is not None:
                    ax.scatter(
                        mean[j],
                        mean[i],
                        c="b",
                        marker="x",
                        s=40,
                        zorder=10
                    )

          
            if i == n_param - 1:
                ax.set_xlabel(param_names[j])
            else:
                ax.set_xticks([])

            if j == 0:
                ax.set_ylabel(param_names[i])
            else:
                ax.set_yticks([])

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    plt.savefig(f"Images/{title}.png", dpi=150, bbox_inches="tight")
    plt.close()
