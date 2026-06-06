import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm


def calculate_mt_response(model, frequencies):
    mu = 4 * np.pi * 1E-7

    n_layers = (len(model) + 1) // 2
    resistivities = model[:n_layers]
    thicknesses = model[n_layers:]

    apparent_resistivities = []
    phases = []

    for frequency in frequencies:
        w = 2 * np.pi * frequency
        impedances = [0] * n_layers
        impedances[-1] = np.sqrt(w * mu * resistivities[-1] * 1j)

        for j in range(n_layers - 2, -1, -1):
            resistivity = resistivities[j]
            thickness = thicknesses[j]
            dj = np.sqrt((w * mu * (1.0 / resistivity)) * 1j)
            wj = dj * resistivity
            ej = np.exp(-2 * thickness * dj)
            belowImpedance = impedances[j + 1]
            rj = (wj - belowImpedance) / (wj + belowImpedance)
            re = rj * ej
            Zj = wj * ((1 - re) / (1 + re))
            impedances[j] = Zj

        Z = impedances[0]
        absZ = abs(Z)
        apparentResistivity = (absZ * absZ) / (mu * w)
        phase = np.atan2(Z.imag, Z.real)
        apparent_resistivities.append(apparentResistivity)
        phases.append(np.rad2deg(phase))

    return np.array(apparent_resistivities + phases).flatten()



data = np.genfromtxt(
    "../MT/CUSTOM_NA.txt",
    delimiter="\t",
    skip_header=1
)

print("\n")

LAYER=int(input("Masukkan jumlah lapisan : "))
maxdepth = int(input("Masukkan kedalaman maksimum (m): "))
init_param = float(input("initial model (ohm.m): "))

frequencies = data[:, 0]
Model = [init_param] * (LAYER * 2 - 1)
ModelCopy = Model.copy()


# 11 RHoA  11 Phs
Observasi = np.concatenate((data[:, 1], data[:, 2]))

# 11 Freq
J= np.zeros((len(frequencies)*2, len(Model)))




Mfit=[]
Iterasi = int(input("Masukkan jumlah iterasi : "))  
for i in tqdm(range(Iterasi), desc="Progress"):
    perturb_factor = 1.01
    Forward = calculate_mt_response(Model, frequencies)
    PerbasiModel=(np.array(Model)*perturb_factor)-np.array(Model)

    perturbed_models = []

    for  j in range(len(Model)):
        model_perturbed = Model.copy()
        model_perturbed[j] = model_perturbed[j] * perturb_factor
        perturbed_models.append(model_perturbed)

    for k in range(len(perturbed_models)):
        J[:, k]=(calculate_mt_response(perturbed_models[k], frequencies)-Forward)/PerbasiModel[k]

    I = np.eye(len(Model))
    Model = Model + (np.linalg.inv(J.T @ J + 0.1**2 * I) @ J.T @ (Observasi - Forward))


    Model[0:] = np.abs(Model[0:])


    missfit = np.sqrt(
                np.sum(
                    (np.log10(Observasi[:len(frequencies)] / Forward[:len(frequencies)]))**2 +
                    ((np.deg2rad(Observasi[len(frequencies):])) - (np.deg2rad(Forward[len(frequencies):])))**2
                )/len(frequencies)
                )*100


    Mfit.append(missfit)


x = np.arange(len(Mfit))

plt.figure(figsize=(8, 5))

plt.plot(x, Mfit, label='Trend Misfit', zorder=1)

plt.scatter(x, Mfit, color="r", s=20, label='Data Poin', zorder=2)

plt.xlabel('Iterasi')
plt.ylabel('Misfit')
plt.title('LM: Misfit vs Iterasi')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()
apparent_resistivities = Observasi[:len(frequencies)]
phases = Observasi[len(frequencies):]
fapparent_resistivities = Forward[:len(frequencies)]
fphases = Forward[len(frequencies):]

periods = 1 / np.array(frequencies)



Initialresistivities = np.array(ModelCopy[:LAYER])
Initialthicknesses = np.array(ModelCopy[LAYER:])
resistivities = np.array(Model[:LAYER])
thicknesses = np.array(Model[LAYER:])



# Kedalaman batas tiap lapisan
depths_init = np.concatenate(([0], np.cumsum(Initialthicknesses)))
depths_init = np.append(depths_init, depths_init[-1] + 11500)  

depths = np.concatenate(([0], np.cumsum(thicknesses)))
depths = np.append(depths, depths[-1] + 11500)

# Buat data untuk plot step - initial
depths_plot_init = np.repeat(depths_init, 2)[1:-1]
resistivities_plot_init = np.repeat(Initialresistivities, 2)

# Buat data untuk plot step - inversi
depths_plot = np.repeat(depths, 2)[1:-1]
resistivities_plot = np.repeat(resistivities, 2)



fig = plt.figure(figsize=(12,7))

gs = GridSpec(
    2, 2,
    width_ratios=[1.6, 0.7],
    height_ratios=[1,1]
)
# =========================
# Apparent Resistivity
# =========================
ax1 = fig.add_subplot(gs[0,0])

ax1.loglog(periods, apparent_resistivities,
           'ko', label='Observed')

ax1.loglog(periods, fapparent_resistivities,
           'r-', linewidth=3, label='Best Fit')
ax1.set_ylim(1e1, 1e3)
ax1.set_xlabel("Period (s)")
ax1.set_ylabel("Apparent Resistivity (Ω·m)")
ax1.set_title("Apparent Resistivity")
ax1.legend()


# =========================
# Phase
# =========================
ax2 = fig.add_subplot(gs[1,0])

ax2.semilogx(periods, phases,
             'ko', label='Observed')

ax2.semilogx(periods, fphases,
             'r-', linewidth=3, label='Best Fit')

ax2.set_xlabel("Period (s)")
ax2.set_ylabel("Phase (deg)")
ax2.set_title("Phase")
ax2.set_ylim(0,90)


# =========================
# Model Resistivity vs Depth
# =========================
ax3 = fig.add_subplot(gs[:,1])

ax3.step(resistivities_plot_init,
         depths_plot_init,
         where='pre',
         color='black',
         linewidth=2,
         label='Initial')

ax3.step(resistivities_plot,
         depths_plot,
         where='pre',
         linestyle='--',
         color='red',
         linewidth=3,
         label='Best Inversion')

ax3.set_xscale('log')
ax3.invert_yaxis()

ax3.set_xlabel("Resistivity (Ω·m)")
ax3.set_ylabel("Depth (m)")
ax3.set_title(f"Model Misfit {missfit:.2f} %")

ax3.set_xlim(1,1e4)
ax3.set_ylim(maxdepth,0)

ax3.legend()


plt.tight_layout()
plt.show()
