import rrcf
import numpy as np


def detectar_outliers_cuartiles(X, factor=1.5):
    Q1 = np.percentile(X, 25, axis=0)
    Q3 = np.percentile(X, 75, axis=0)
    IQR = Q3 - Q1

    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR

    outliers = (X < lower_bound) | (X > upper_bound)

    return np.any(outliers, axis=1).astype(int)


def detectar_outliers_zscore(X, threshold=3):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    z_scores = np.abs((X - mean) / std)
    outliers = z_scores > threshold
    return np.any(outliers, axis=1).astype(int)


def detectar_outliers_mad(X, threshold=2):
    median = np.median(X, axis=0)
    mad = np.median(np.abs(X - median), axis=0)
    scaled_mad = np.abs((X - median) / mad)
    outliers = scaled_mad > threshold
    return np.any(outliers, axis=1).astype(int)

def detectar_outliers_rrcf(X, d=3, num_trees=100, tree_size=256):
    """
    Calculate average CoDisp.

    Parameters:
        X (array-like): Data array.
        d (int): Dimension of the data.
        num_trees (int): Number of trees to build in the forest.
        tree_size (int): Number of points to include in each tree.

    Returns:
        pd.Series: Average CoDisp values for each point in the data.
    """
    n = len(X)

    # Construct forest
    forest = []
    while len(forest) < num_trees:
        # Select random subsets of points uniformly from point set
        ixs = np.random.choice(n, size=(n // tree_size, tree_size),
                               replace=False)
        # Add sampled trees to forest
        trees = [rrcf.RCTree(X[ix], index_labels=ix) for ix in ixs]
        forest.extend(trees)

    # Compute average CoDisp
    avg_codisp = pd.Series(0.0, index=np.arange(n))
    index = np.zeros(n)
    for tree in forest:
        codisp = pd.Series({leaf: tree.codisp(leaf) for leaf in tree.leaves})
        avg_codisp[codisp.index] += codisp
        np.add.at(index, codisp.index.values, 1)
    avg_codisp /= index

    return avg_codisp