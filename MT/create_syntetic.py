import numpy as NP
from MT_function import forward

Model = NP.array([
    10, 1000, 100, 300, 600   
    ])

frequencies = NP.array([
    2., 2.594773, 3.366423, 4.367552, 5.666402, 7.351514,
    9.537754, 12.374153, 16.054058, 20.828317, 27.022376,
    35.058464, 45.484377, 59.010813, 76.559829, 99.327684,
    128.86639, 167.189507, 216.909399, 281.415312,
    365.10441, 473.68151, 614.547968, 797.306199,
    1034.41425, 1342.03502, 1741.13804, 2258.92888,
    2930.70369, 3802.25523, 4932.99437, 6400.
])

#=======================================================================================================
if len(Model) % 2 == 0 :
    print("Total parameter must be odd number (resistivity + thickness) ")
    exit()

noise_level = int(input("Noise Level (%) ..? "))

n_params = len(Model)
n_layer = (n_params + 1) // 2  # (5 + 1) // 2 = 3


_, Obsres, Obsphs = forward(Model, frequencies, 0, 0)


Obsres += NP.random.uniform(-Obsres/100*noise_level, Obsres/100*noise_level)
Obsphs += NP.random.uniform(-Obsphs/100*noise_level, Obsphs/100*noise_level)

Obsres = Obsres.squeeze()
Obsphs = Obsphs.squeeze()

mt_data = NP.column_stack((frequencies, Obsres, Obsphs))

NP.savetxt(f'MT/synthetic_{n_layer}_layer.txt', 
           mt_data,
           header=f'{NP.array2string(Model, separator="\t")[1:-1]}\nFrequency(Hz)\tAppRes(Ohm.m)\tPhase(deg)', 
           delimiter='\t',
           comments='') 

print("\n")
print("=====> Success!")
