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

def compute_hks(filename, args):
    name = os.path.splitext(filename)[0]
    if os.path.exists(name + '.npz'):
        extractor = signature.SignatureExtractor(path=name+'.npz')
    else:
        mesh = trimesh.load(filename)
        extractor = signature.SignatureExtractor(mesh, 100, args.approx)
        np.savez_compressed(name+'.npz', evals=extractor.evals, evecs=extractor.evecs)
    
    return extractor.heat_signatures(100)

parser = argparse.ArgumentParser(description='Mesh signature visualization')
parser.add_argument('--file', default='models/cat0.off', type=str, help='3D Model used')
parser.add_argument('--n_basis', default='100', type=int, help='Number of basis used')
parser.add_argument('--f_size', default='128', type=int, help='Feature size used')
parser.add_argument('--approx', default='cotangens', choices=laplace.approx_methods(), type=str, help='Laplace approximation to use')
parser.add_argument('--laplace', help='File holding laplace spectrum')
parser.add_argument('--kernel', type=str, default='heat', help='Feature type to extract. Must be in [heat, wave]')

args = parser.parse_args()

file = args.file

vids = [18047, 14680, 25201, 21688]

hks = compute_hks(file, args)

plt.plot(hks[vids[0],:], color='red')
plt.plot(hks[vids[1],:], color='red')
plt.plot(hks[vids[2],:], color='blue')
plt.plot(hks[vids[3],:], color='blue')

plt.show()

# Cargar la malla para Polyscope
mesh = trimesh.load(file)

# Inicializar Polyscope
ps.init()

# Registrar la malla en Polyscope
ps_mesh = ps.register_surface_mesh("3D Model", mesh.vertices, mesh.faces)

# Obtener el valor de calor en el último tiempo para cada vértice
heat_last_time = hks[:, 50]

# Asignar los valores de calor como un atributo escalar en Polyscope
ps_mesh.add_scalar_quantity("Heat Signature (last time)", heat_last_time, defined_on='vertices', cmap='coolwarm')

# Mostrar la visualización
ps.show()
