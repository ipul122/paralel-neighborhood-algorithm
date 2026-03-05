import numpy as NP
from function import create_bounds

nr = 10
ns = 150

ni = 150
iter = 550

seed = None




def type_banchmark(type_input=1):

    
    if type_input == 1:
        print("Rosenbrock Function")
        lb = NP.array([-2, -1])
        ub = NP.array([2, 3])
        
    elif type_input == 2:
        print("MT 1D Function")
        nlayer= int(input("Nlayer (syntetic 3 / 4 / 5 /) (6 custom)..? "))
        print("\n")

        if nlayer in [3, 4, 5]:
            model, data = NP.loadtxt(f'./MT/synthetic_{nlayer}_layer.txt', max_rows=1), NP.loadtxt(f'./MT/synthetic_{nlayer}_layer.txt', skiprows=2)
            
            frequencies, Obsres, Obsphs = data[:, 0], data[:, 1], data[:, 2]

            ResReal = NP.array(model[:nlayer])
            ThkReal = NP.array(model[nlayer:])
            lb, ub = create_bounds(nlayer)

        elif nlayer == 6:
            
            layer = int(input(" custom layer 3-10..? "))
            
            if layer in [3, 4, 5, 6, 7, 8, 9, 10]:
                data = NP.loadtxt(f'./MT/CUSTOM.txt', skiprows=1)
                frequencies, Obsres, Obsphs = data[:, 0], data[:, 1], data[:, 2]

                lb, ub = create_bounds(layer)
                ResReal = NP.zeros(nlayer)
                ThkReal = NP.zeros(nlayer-1)    
                maxdepth = int(input(" maxdepth > 1000 m ? "))

                return lb, ub, frequencies, Obsres, Obsphs, ResReal, ThkReal, layer, maxdepth
            else : 
                print("ngopi dulu !")

        else:
            print(f"nlayer {nlayer} tidak didukung")
            exit()

        return lb, ub, frequencies, Obsres, Obsphs, ResReal, ThkReal, nlayer, 2000
        
    else:
        print("Tidak dikenali !!")
        exit()

    return lb, ub, 0, 0, 0, 0, 0, 0, 0