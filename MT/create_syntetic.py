import numpy as NP
from MT_function import forward

Model = NP.array([
    	  200,	  10,	1000,	 300,	 600

    ]).reshape(1, -1)

frequencies = NP.array([
    1.0000000000000000e+02,
    5.6234130000000000e+01,
    3.1622779999999999e+01,
    1.7782789999999999e+01,
    1.0000000000000000e+01,
    5.6234130000000002e+00,
    3.1622780000000001e+00,
    1.7782789999999999e+00,
    1.0000000000000000e+00,
    5.6234099999999998e-01,
    3.1622800000000001e-01,
    1.7782800000000001e-01,
    1.0000000000000001e-01,
    5.6233999999999999e-02,
    3.1622999999999998e-02,
    1.7783000000000000e-02,
    1.0000000000000000e-02,
    5.6230000000000004e-03,
    3.1619999999999999e-03,
    1.7780000000000001e-03,
    1.0000000000000000e-03
])


#=======================================================================================================
if len(Model) % 2 == 0 :
    print("Total parameter must be odd number (resistivity + thickness) ")
    exit()

noise = int(input("Noise Level (%) ..? "))

n_params = len(Model[0])
n_layer = (n_params + 1) // 2  # (5 + 1) // 2 = 3

Obsres_dummy = NP.ones(len(frequencies))      
Obsphs_dummy = NP.zeros(len(frequencies))     
_, Obsres, Obsphs = forward(Model, frequencies, Obsres_dummy  , Obsphs_dummy, noise_level=noise)
_, ObsresClean, ObsphsClean = forward(Model, frequencies, Obsres_dummy  , Obsphs_dummy)

Obsres = Obsres.squeeze()
Obsphs = Obsphs.squeeze()
ObsresClean = ObsresClean.squeeze()
ObsphsClean = ObsphsClean.squeeze()

mt_data = NP.column_stack((frequencies, Obsres, Obsphs))

NP.savetxt(f'MT/synthetic_{n_layer}_layer.txt', 
           mt_data,
           header=f'{NP.array2string(Model.flatten(), separator="\t")[1:-1]}\nFrequency(Hz)\tAppRes(Ohm.m)\tPhase(deg)', 
           delimiter='\t',
           comments='') 

print("\n")
print("=====> Success!")


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(8, 10), dpi=120)
gs = gridspec.GridSpec(2, 1, height_ratios=[4, 2])

ax0 = fig.add_subplot(gs[0]) 
ax1 = fig.add_subplot(gs[1]) 

ax0.loglog(1/frequencies, ObsresClean, "r-", linewidth=4, label="original")
ax0.loglog(1/frequencies, Obsres, "ko", markersize=10, label="noise")
ax0.set_ylabel("Apparent Resistivity (Ohm.m)")
ax0.set_title("Apparent Resistivity")
ax0.set_ylim(1e-0, 1e4)
ax0.legend(fontsize=8)

ax1.semilogx(1/frequencies, ObsphsClean, "r-", linewidth=4)
ax1.semilogx(1/frequencies, Obsphs, "ko", markersize=10)
ax1.set_xlabel("Period (s)")
ax1.set_ylabel("Phase (deg)")
ax1.set_ylim(0, 90)

plt.tight_layout()
fig.savefig("MT/anu.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)