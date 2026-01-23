import numpy as NP
from function import rosenbrock, plot_all_models, plot_rosenbrock, plot_obs, corner_plot
from NA.Search import sampling, dim_models, generate_random_models
from config import nr, ns, ni, iter, type_banchmark
from MT.MT_function import forward
from tqdm import tqdm
from NA.Appraise import NAAppraiser
from NA.SEARCH_COBA import sampling_jit


nss= ns
type = int(input("1. rosenbrock, 2. MT1D..? "))  


lb, ub, frequencies, Obsres, Obsphs, ResReal, ThkReal , n , maxdepth= type_banchmark(type)

nd = lb.size
ne = ns * (iter - 1) + ni


while True:

    if type != 1:
        ERes = NP.zeros((ne,len(frequencies)))        
        EPhs = NP.zeros((ne,len(frequencies)))     

    models = NP.zeros((ne, nd))
    misfits = NP.zeros(ne)        

    np = 0       
    idx = 0
            
    for it in tqdm(range(iter), desc="Progress"):
        if it == 0:
            ns = ni 
            batch = generate_random_models(ni,nd)
            background = lb + (ub - lb) * generate_random_models(50000,nd)
            background_misfits = rosenbrock(background)

        else:
            ns = nss 
<<<<<<< HEAD
            batch = sampling(ns,
=======
            batch = sampling_jit(ns,
>>>>>>> e362550 (Final push all files)
                            nd,
                            nr,
                            models,
                            np,
<<<<<<< HEAD
                            misfits)                                 
        
=======
                            misfits)
                                    
>>>>>>> e362550 (Final push all files)
        models[idx:idx+ns] = batch
        batch = lb + (ub - lb) * batch

        if type == 1:
            misfits[idx:idx+ns] = rosenbrock(batch)
        else:
            misfits[idx:idx+ns],ERes[idx:idx+ns],EPhs[idx:idx+ns] = forward(batch, frequencies, Obsres, Obsphs)

        idx += ns
        np += ns
       
    best_idx = NP.argmin(misfits)
    models = dim_models(lb, ub, models)
    best_model = models[best_idx]
    print("Best model:", best_model)
    print(f"Best misfit: {misfits[best_idx]:.3f}")

    if type == 1:
        plot_rosenbrock(models, best_model, background, background_misfits, save_path=f"Images/rosenbrock.png")

    else: 
        Resis = best_model[:n]  
        Thick = best_model[n:]
        bestmisfit, res, phs = forward(best_model.reshape(1,-1), frequencies, Obsres, Obsphs) 
        plot_obs(frequencies, Obsres, Obsphs, res, phs, ERes, EPhs,save_path=f"Images/curve_{n}layer.png")
        plot_all_models(models,Resis,Thick,ResReal,ThkReal,n,save_path=f"Images/model_{n}layer.png", depthmax=maxdepth)



    """if  misfits[best_idx]< 9:
        break"""

    repeat = input("Repeat program? (y/n): ")
    
    if repeat.lower() != 'y':
        break  

<<<<<<< HEAD


exit()

=======
>>>>>>> e362550 (Final push all files)
results = NAAppraiser(
    initial_ensemble=models,
    log_ppd= -misfits,
    bounds = tuple(zip(lb, ub)),
    n_resample=10000,
    n_walkers=8
)

results.run()


if type ==1 :
    true_model = NP.array([1,1])
else : true_model = NP.hstack((ResReal, ThkReal))


corner_plot(
    type=type,
    samples=results.samples,
    mean=results.mean,
    best_model=best_model,
    true_model=true_model,
)




"""
import matplotlib.pyplot as plt
# Hanya p(x|y=1) saja
plt.figure(figsize=(8, 5))

# Conditional posterior samples p(x|y=1)
y_hist_edges = NP.histogram_bin_edges(results.samples[:, 1], bins=50, range=(-1, 3))
best_ind_y = NP.digitize(1, y_hist_edges)

# Filter data di mana y ≈ 1 (dalam satu bin)
x_given_y = results.samples[
    (results.samples[:, 1] > y_hist_edges[best_ind_y - 1]) &
    (results.samples[:, 1] < y_hist_edges[best_ind_y]),
    0
]

plt.hist(x_given_y, bins=25, color="steelblue", alpha=0.7, edgecolor='black')
plt.axvline(1, c="r", ls="--", lw=2, label="x = 1")
plt.xlabel("Nilai X")
plt.ylabel("Frekuensi")
plt.title(f"Histogram")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"Images/histogram_COBAparameter.png", dpi=150, bbox_inches="tight")
plt.close()

"""