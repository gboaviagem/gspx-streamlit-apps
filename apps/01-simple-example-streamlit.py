"""Simple example on first uses of QGSP."""
"""Basic spectral analysis using Quaternion Graph Signal Processing (QGSP)."""
import sys
import os
import pathlib
pdir = pathlib.Path(__file__).parent.resolve()

# Adding the gspx folder to the Python path
sys.path.insert(0, str(pdir).split("gspx")[0] + "/gspx")

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from gspx.datasets import WeatherGraphData
from gspx.signals import QuaternionSignal
from gspx.adaptive import QLMS
from gspx.qgsp import QGFT, QMatrix
from gspx.utils.display import plot_quaternion_graph_signal, plot_graph
from io import BytesIO


def show_and_save(
        plot_name: str, fig = None, save_to: str = "/figures",
        dpi: int = None, fig_size_inches: tuple = None,
        img_format: str = "pdf"):
    """Show the figure in Streamlit and saves locally."""
    if fig is None:
        fig = plt.gcf()
    if fig_size_inches is not None:
        fig.set_size_inches(fig_size_inches[0], fig_size_inches[1])
    if dpi is not None:
        fig.set_dpi(dpi)
    buf = BytesIO()
    fig.savefig(buf, format="png")
    st.image(buf)

    try:
        path = "./" + save_to.lstrip("/").rstrip("/")
        if not os.path.exists(path):
            os.makedirs(path)
        fig.savefig(f'{path}/{plot_name}.{img_format}', dpi=300)
    except Exception as e:
        print(f"Unable to save figure `{plot_name}`: ", e)


st.markdown(
    """
    #### ***QGSP in action***

    # Toy example on quaternion adjaceny matrices and QGFT

    This streamlit app aims to

    - Illustrate the creation of quaternion matrices.
    - Show how to plot a graph.
    - Show how to compute the QGFT and show the eigenvectors total variation.

    Let us first build a graph connecting some cities in England, using edges with quaternion-valued weights and define over it a quaternion-valued signal.
    """
)

# If gspx is not installed, we add it to the path
import os, sys
gdir = os.getcwd()
sys.path.insert(0, gdir)

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from pyquaternion import Quaternion

from gspx.utils.display import plot_graph
from gspx.qgsp import QMatrix
from gspx.utils.quaternion_matrix import \
    explode_quaternions, implode_quaternions
from gspx.qgsp import QGFT

# Let us say we have the coordinates of some graph vertices
coords = np.array([
    [0, 1],
    [-1, 0],
    [0, -1],
    [1, 0],
    [0, 0],
])

# Indices of non-zero adjacency matrix entries
idx = np.array([
    [0, 4],
    [1, 4],
    [2, 4],
    [3, 4],
    [3, 0]
])

# Filling the adjacency matrix
Aq = QMatrix.from_matrix(implode_quaternions(np.zeros((5, 5, 4))))
Ne = 5
rnd = np.random.RandomState(seed=42)
entries = rnd.randint(10, size=(Ne, 4))

for n, i in enumerate(idx):
    Aq.matrix[i[0], i[1]] = Quaternion(entries[n])

# Symmetric graph
Aq = Aq + Aq.conjugate().transpose()

eigq, Vq = Aq.eigendecompose(hermitian_gso=True)

print("Adjacency matrix rows:")
for row in Aq.matrix:
    print([str(q) for q in row])

# Visualizing the graph via Networkx
A_ = Aq.abs()
xi, yi = np.where(A_)
edgelist = [
    (xi[n], yi[n], {'weight': A_[xi[n], yi[n]]})
    for n in range(len(xi))
]

g = nx.DiGraph()
g.add_edges_from(edgelist)
plot_graph(
    g, coords, figsize=(3, 3), node_size=60,
    edge_color=(0.8, 0.8, 0.8, 0.8))

# Total variation of eifenvectors
qgft2 = QGFT(norm=1)
qgft2.fit(Aq)

plt.figure(figsize=(4, 2))
plt.scatter(np.real(qgft2.eigc), np.imag(qgft2.eigc), c=qgft2.tv_)
plt.colorbar()
plt.title("Total Variation of eigenvectors for each eigenvalue")
plt.xlabel("Real(eigvals)")
plt.ylabel("Imag(eigvals)")
plt.tight_layout()
plt.show()
