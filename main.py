import numpy as np
from scipy.sparse import *
from scipy.sparse.linalg import eigs
import openmesh
import argparse
import os
import polyscope as ps
import trimesh

import signature
import laplace
import matplotlib.pyplot as plt

def compute_signatures(filename, n_basis, dim, approx, reload):
    name = os.path.splitext(filename)[0]
    if os.path.exists(name + '.npz') and not reload:
        extractor = signature.SignatureExtractor(path=name+'.npz')
    else:
        mesh = trimesh.load(filename)
        extractor = signature.SignatureExtractor(mesh, n_basis, approx)
        np.savez_compressed(name+'.npz', evals=extractor.evals, evecs=extractor.evecs)
    gps = extractor.evecs[:,1:n_basis ]/np.sqrt(np.tile(extractor.evals[1:n_basis], (extractor.evecs.shape[0],1)))
    print(gps.shape)
    return extractor.evals, extractor.heat_signatures(dim), gps

parser = argparse.ArgumentParser(description='Mesh signature visualization')
parser.add_argument('--n_basis', default='100', type=int, help='Number of basis used')
parser.add_argument('--dim', default='100', type=int, help='Time steps dimension')
parser.add_argument('--time', default='50', type=int, help='Time step to show')
parser.add_argument('--reload', action='store_true', help='Force recalculations')
parser.add_argument('--approx', default='cotangens', choices=laplace.approx_methods(), type=str, help='Laplace approximation to use')
parser.add_argument('--laplace', help='File holding laplace spectrum')
parser.add_argument('--t_sp', default='90', type=int, help='Special time')
args = parser.parse_args()

def plot_descriptors(models, n_basis, dim, approx, time, reload, t_sp):
    """Calcula y grafica el descriptor HKS y GPS de varios modelos en dos gráficos separados."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # Dos gráficos en la misma figura, lado a lado

    for model in models:
        evals, hks, gps = compute_signatures(model, n_basis, dim, approx, reload)

        # Identificar los vértices de interés
        mesh = trimesh.load(model)
        adjacency_list = mesh.vertex_adjacency_graph
        target_vertices = []

        for v in range(len(mesh.vertices)):
            neighbors = adjacency_list.neighbors(v)
            if all(hks[v, t_sp] > hks[n, t_sp] for n in neighbors):
                target_vertices.append(v)

        interval_descriptors = hks[target_vertices, :]
        vertex_averages = np.mean(interval_descriptors, axis=0)  

        gps_vertex_averages = np.mean(gps[target_vertices, :], axis=0)  

        x_range_hks = range(len(vertex_averages))
        x_range_gps = range(len(gps_vertex_averages))

        color = "blue" if "cat" in model else "green"
        # Gráfico de HKS
        axes[0].plot(x_range_hks, vertex_averages, label=f"HKS {model}", color=color)
        axes[0].set_title("Promedios del descriptor HKS")
        axes[0].set_xlabel("Índice del valor propio")
        axes[0].set_ylabel("Valor promedio")
        axes[0].legend()
        axes[0].grid(True)

        # Gráfico de GPS
        axes[1].plot(x_range_gps, gps_vertex_averages, label=f"GPS {model}", color=color)
        axes[1].set_title("Promedios del descriptor GPS")
        axes[1].set_xlabel("Índice del valor propio")
        axes[1].set_ylabel("Valor promedio")
        axes[1].legend()
        axes[1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Parámetros del HKS
    n_basis = args.n_basis  # Número de valores propios
    dim = args.dim      # Dimensiones de tiempo
    approx = args.approx  # Aproximación de Laplace
    time = args.time # Timestep
    reload = args.reload # new npz
    t_sp = args.t_sp # Tiempo especial
   
    models = ["models/cat" + str(i) + ".off" for i in range(11)]
    
    for i in range(18):
        if os.path.exists("models/horse" + str(i) + ".off"):
            models += ["models/horse" + str(i) + ".off"]
    # Genera y muestra el gráfico
    plot_descriptors(models, n_basis, dim, approx, time, reload, t_sp)




    """
    Para ejecutar el código basta con colocar "python main.py". Se agregaron otros parámetros para estudiar el comportamiento nada más.

    Preguntas del Enunciado:  
    ¿Qué podrías concluir de los descriptores resultantes? ¿Cuál de las dos opciones es mejor para describir objetos de manera invariante a
    transformaciones no rígidas?

    Si bien no se descarta la posible utilidad de GPS como descriptor de modelos, para comportamientos ante transformaciones no rígidas 
    HKS muestra resultados visuales, "legibles" y numéricamente consistentes, haciéndolo la mejor opción como descriptor de objetos. Vemos 
    que para los mismos índices de valor propio la curva descrita es similar para modelos de un mismo objeto pero al mismo tiempo lo diferencia
    de otros objetos. 

    """