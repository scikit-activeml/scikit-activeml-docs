"""
Regression Tree Based Active Learning (RT-AL) with Representativity Selection
=============================================================================
"""

# %%
# 

# %%
# .. note::
#    The generated animation can be found at the bottom of the page.
#
# | **Google Colab Note**: If the notebook fails to run after installing the
#   needed packages, try to restart the runtime (Ctrl + M) under
#   Runtime -> Restart session.
#
# .. image:: https://colab.research.google.com/assets/colab-badge.svg
#    :target: https://colab.research.google.com/github/scikit-activeml/scikit-activeml-docs/blob/gh-pages/latest/generated/sphinx_gallery_notebooks//pool/plot-RegressionTreeBasedAL-Regression_Tree_Based_Active_Learning_(RT-AL)_with_Representativity_Selection.ipynb
#
# | **Notebook Dependencies**
# | Uncomment the following cell to install all dependencies for this
#   tutorial.

# !pip install scikit-activeml

# %%
#.. raw:: html
#
#   <hr style="border-style: solid; border-top: 1px solid; border-right: 0; border-bottom: 0; border-left: 0;">
#

# %%
import numpy as np
from matplotlib import pyplot as plt, animation
from scipy.stats import uniform

from skactiveml.utils import MISSING_LABEL, labeled_indices, is_labeled

from sklearn.tree import DecisionTreeRegressor
from skactiveml.regressor import SklearnRegressor
from skactiveml.pool import RegressionTreeBasedAL

random_state = np.random.RandomState(0)


def true_function(X_):
    return (X_**3 + 2 * X_**2 + X_ - 1).flatten()

n_samples = 100
X = np.concatenate(
        [uniform.rvs(0, 1.5, 9 * n_samples // 10, random_state=random_state),
         uniform.rvs(1.5, 0.5, n_samples // 10, random_state=random_state)]
    ).reshape(-1, 1)

noise = np.vectorize(lambda x : random_state.rand() * 1.5 if x < 1
                                else random_state.rand() * 0.5)

# Build a dataset.
y_true = true_function(X) + noise(X).flatten()
y = np.full(shape=y_true.shape, fill_value=MISSING_LABEL)
X_test = np.linspace(0, 2, num=1000).reshape(-1, 1)

# Initialise the classifier.
reg = SklearnRegressor(DecisionTreeRegressor(min_samples_leaf=2, random_state=random_state), random_state=random_state)
# Initialise the query strategy.
qs = RegressionTreeBasedAL(method='representativity', random_state=random_state)


# Preparation for plotting.
fig, (ax_1, ax_2) = plt.subplots(2, 1, sharex=True)
artists = []

# The active learning cycle:
n_cycles = 20
batch_size = 3
for c in range(n_cycles):
    # Fit the classifier.
    reg.fit(X, y)

    # Get labeled instances.
    X_labeled = X[labeled_indices(y)]

    # Query the next instance/s.
    query_idx, utilities = qs.query(X=X, y=y, reg=reg, return_utilities=True, batch_size=batch_size)

    # Plot the labeled data.
    coll_old = list(ax_1.collections) + list(ax_2.collections)
    title = ax_1.text(
        0.5,
        1.05,
        f"Prediction after acquring {c} labels",
        size=plt.rcParams["axes.titlesize"],
        ha="center",
        transform=ax_1.transAxes,
    )

    sort_mask = np.argsort(X.flatten())
    X_plot = X.flatten()[sort_mask]
    utilities_plot = utilities[0][sort_mask]/batch_size

    is_lbld = is_labeled(y)

    (utility_line,) = ax_2.plot(X_plot, utilities_plot, c="green")
    utility_fill = plt.fill_between(
        X_plot, utilities_plot, color="green", alpha=0.3
    )

    ax_1.scatter(X[~is_lbld], y_true[~is_lbld], c="lightblue")
    ax_1.scatter(X[is_lbld], y[is_lbld], c="orange")

    y_pred = reg.predict(X_test)
    (prediction_line,) = ax_1.plot(X_test, y_pred, c="black")

    coll_new = list(ax_1.collections) + list(ax_2.collections)
    coll_new.append(title)
    artists.append(
        [x for x in coll_new if (x not in coll_old)]
        + [utility_line, utility_fill, prediction_line]
    )

    # Label the queried instances.
    y[query_idx] = y_true[query_idx]

ani = animation.ArtistAnimation(fig, artists, interval=1000, blit=True)

# %%
# .. image:: ../../examples/pool_regression_legend.png

# %%
# .. rubric:: References:
# 
# The implementation of this strategy is based on :footcite:t:`jose2023regression`.
#
# .. footbibliography::


