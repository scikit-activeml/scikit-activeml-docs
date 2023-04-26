"""
Batch Bayesian Active Learning by Disagreement (BatchBALD)
==========================================================
"""

# %%
# 

import numpy as np
from matplotlib import pyplot as plt, animation
from sklearn.datasets import make_blobs

from skactiveml.utils import MISSING_LABEL, labeled_indices, unlabeled_indices
from skactiveml.visualization import plot_utilities, plot_decision_boundary, \
    plot_contour_for_samples

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import BaggingClassifier
from skactiveml.classifier import SklearnClassifier
from skactiveml.pool import BatchBALD

random_state = np.random.RandomState(0)

# Build a dataset.
X, y_true = make_blobs(n_samples=200, n_features=2,
                       centers=[[0, 1], [-3, .5], [-1, -1], [2, 1], [1, -.5]],
                       cluster_std=.7, random_state=random_state)
y_true = y_true % 2
y = np.full(shape=y_true.shape, fill_value=MISSING_LABEL)

# Initialise the classifier.
clf = SklearnClassifier(BaggingClassifier(
    SklearnClassifier(GaussianProcessClassifier(), random_state=random_state), 
    random_state=random_state),
    classes=np.unique(y_true),
    random_state=random_state
)
# Initialise the query strategy.
qs = BatchBALD(random_state=random_state)


# Preparation for plotting.
fig, axs = plt.subplots(2, 2, constrained_layout=True)
feature_bound = [[min(X[:, 0]), min(X[:, 1])], [max(X[:, 0]), max(X[:, 1])]]
artists = [[] for j in range(5)]

# The active learning cycle:
n_cycles = 5
for c in range(n_cycles):
    # Fit the classifier.
    clf.fit(X, y)

    # Get labeled instances.
    X_labeled = X[labeled_indices(y)]

    # Query the next instance/s.
    query_idx, utilities = qs.query(X=X, y=y, ensemble=clf, batch_size=4, return_utilities=True)

    # Plot the labeled data.
    for i, ax in enumerate(axs.flatten()):
        coll_old = list(ax.collections)
        plot_contour_for_samples(X, utilities[i], res=25,
                                 feature_bound=feature_bound, replace_nan=None, ax=ax)
        ax.scatter(X[:, 0], X[:, 1], c=y_true, cmap="coolwarm", marker=".",
                   zorder=2)
        ax.scatter(X_labeled[:, 0], X_labeled[:, 1], c="grey", alpha=.8,
                   marker=".", s=300)
        ax = plot_decision_boundary(clf, feature_bound, ax=ax)
        ax.set_title(f"Batch {c+1}, Utilities[{i}]")

        for x in ax.collections:
            if x not in coll_old:
                artists[c].append(x)

    # Label the queried instances.
    y[query_idx] = y_true[query_idx]

ani = animation.ArtistAnimation(fig, artists, interval=1000, blit=True)

# %%
# .. image:: ../../examples/pool_classification_legend.png

# %%
# .. rubric:: References:
# 
# The implementation of this strategy is based on :footcite:t:`houlsby2011bayesian` and :footcite:t:`kirsch2019batchbald`.
#
# .. footbibliography::


