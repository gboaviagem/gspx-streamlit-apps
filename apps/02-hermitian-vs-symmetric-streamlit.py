"""The QGFT in Hermitian and symmetric graphs."""
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

    # The QGFT in Hermitian vs. Symmetric adjacency matrices

    Where does weather data graph signal is smoother? In a graph with Hermitian ou symmetric quaternion-valued adjacency matrix? Let us see.
    """
)

data = WeatherGraphData(n_neighbors=7, england_only=True)
Ar, coords = data.graph
s = data.signal
df = data.data

st.markdown("The graph signal will have quaternion components (1, i, j, k) given by the columns `'humidity', 'pressure', 'temp', 'wind_speed'` in the following dataframe:")

st.dataframe(data=df, width=1000, height=None)

st.markdown("Let us create a graph with ***Hermitian*** quaternion-valued adjacency matrix:")
st.code(
"""
from gspx.datasets import WeatherGraphData

data = WeatherGraphData(n_neighbors=7, england_only=True)
Ar, coords = data.graph
s = data.signal
df = data.data

Aq = data.quaternion_adjacency_matrix(hermitian=True)
""",
language="python"
)

st.markdown("The data in columns `'humidity', 'pressure', 'temp', 'wind_speed'` is normalized column-wise using linear scaling to make each column fit the range [0, 1]. The graph signal can be visualized by splitting each quaternion component in a pseudocolor scale as such:")

st.markdown("**Quaternion graph signal:**")
fig = plot_quaternion_graph_signal(
    s, coords, figsize=(5, 8)
)
show_and_save(
    plot_name="uk_graph_signal", fig=fig,
    fig_size_inches=(4.5, 5.5))

st.markdown("### Hermitian graph")
Aq = data.quaternion_adjacency_matrix(hermitian=True)

st.markdown(
    """
    The QGFT is computed by the `qgft = QGFT()` object. The eigenvalues and always complex-valued and are stored in the attribute `qgft.eigc`. The associated eigenvectors have their total variation stored in the attribute `qgft.tv_`.
    """
)

qgft = QGFT(norm=1)
qgft.fit(Aq)

plt.figure(figsize=(5, 5), dpi=100)
plt.scatter(np.real(qgft.eigc), np.imag(qgft.eigc), c=qgft.tv_)
plt.colorbar()
plt.title("Total Variation of eigenvectors for each eigenvalue")
plt.xlabel("Real(eigvals)")
plt.ylabel("Imag(eigvals)")

st.markdown("Since this graph was created with a Hermitian adjacency matrix, its eigenvalues are always real-valued:")

show_and_save(
    plot_name="uk_total_variation_hermitian",
    fig_size_inches=(6, 3.5))

st.markdown(
    """
    The spectrum of the quaternion graph signal `s` is obtained by the method `qgft.transform(s)`. The following code illustrates is computation and the spectrum visual representation:
    """
)
st.code(
"""
from gspx.signals import QuaternionSignal

ss = qgft.transform(s)
QuaternionSignal.show(ss, ordering=qgft.idx_freq)
""",
language="python"
)

ss = qgft.transform(s)
fig = QuaternionSignal.show(ss, ordering=qgft.idx_freq)

show_and_save(
    plot_name="uk_spectrum_hermitian", fig=fig,
    fig_size_inches=(9, 5.5))


st.markdown("### Symmetric graph")
Aq = data.quaternion_adjacency_matrix(hermitian=False)

st.markdown(
    """
    The QGFT is computed by the `qgft = QGFT()` object. The eigenvalues and always complex-valued and are stored in the attribute `qgft.eigc`. The associated eigenvectors have their total variation stored in the attribute `qgft.tv_`.
    """
)

qgft = QGFT(norm=1)
qgft.fit(Aq)

plt.figure(figsize=(5, 5), dpi=100)
plt.scatter(np.real(qgft.eigc), np.imag(qgft.eigc), c=qgft.tv_)
plt.colorbar()
plt.title("Total Variation of eigenvectors for each eigenvalue")
plt.xlabel("Real(eigvals)")
plt.ylabel("Imag(eigvals)")

st.markdown("Compare the eigenvectors' total variation with those in the Hermitian graph")

show_and_save(
    plot_name="uk_total_variation_symmetric",
    fig_size_inches=(6, 3.5))

st.markdown(
    """
    The spectrum of the quaternion graph signal in this symmetric graph will finally answer the initial question:
    """
)

ss = qgft.transform(s)
fig = QuaternionSignal.show(ss, ordering=qgft.idx_freq)

show_and_save(
    plot_name="uk_spectrum_symmetric", fig=fig,
    fig_size_inches=(9, 5.5))

h_ideal = np.zeros(len(qgft.idx_freq))
# Bandwith of 20% the frequency support
bandwidth = int(len(qgft.idx_freq) / 5)
h_ideal[qgft.idx_freq[:bandwidth]] = 1

h_idealq = QuaternionSignal.from_rectangular(np.hstack((
    h_ideal[:, np.newaxis],
    np.zeros(len(qgft.idx_freq))[:, np.newaxis],
    np.zeros(len(qgft.idx_freq))[:, np.newaxis],
    np.zeros(len(qgft.idx_freq))[:, np.newaxis]
)))

@st.cache
def qlms_computation():
    N = 7
    X = QMatrix.vander(qgft.eigq, N, increasing=True)
    y = h_idealq

    qlms = QLMS(
        step_size=[0.0001, 0.0003, 0.0005, 0.0006], verbose=2)
    qlms.fit(X, y)
    return X, qlms

with st.spinner('QLMS computation:'):
    X, qlms = qlms_computation()

st.markdown("The plot below shows the cost per iteration of the QLMS (only for the values of step sizes that did not cause interruption by divergence - our implementation has an `early stop` parameter that interrupts the calculations if the cost increases for more than 10 iterations).")
qlms.plot(nsamples=100)

show_and_save(
    plot_name="uk_qlms_iterations_symmetric",
    fig_size_inches=(5.5, 4.5))

st.markdown("The filter taps obtained by the end of the optimization are:")
st.text(str(qlms.res_[qlms.best_lr_]['result']))

st.markdown(
    """
    ### LSI filter response

    The LSI filter obtained using QLMS has the following filter response:
    """
)

h_opt = qlms.predict(X)
h_opt = QuaternionSignal.from_samples(h_opt.matrix.ravel())
fig = QuaternionSignal.show(h_opt, ordering=qgft.idx_freq)
show_and_save(
    plot_name="uk_qlm_filter_symmetric", fig=fig,
    fig_size_inches=(9, 5.5))