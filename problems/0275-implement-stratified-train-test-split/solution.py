import numpy as np

def stratified_train_test_split(X, y, test_size, random_seed=None):
    """
    Split data into train and test sets while maintaining class proportions.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Label vector of shape (n_samples,)
        test_size: Proportion of data for test set (0 < test_size < 1)
        random_seed: Random seed for reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    np.random.seed(random_seed)
    X_train = np.array([])
    X_test = np.array([])
    y_train = np.array([])
    y_test = np.array([])
    unique = np.unique(y)
    for ele in unique:
        X_temp = X[y == ele]
        y_temp = y[y == ele]
        train_idx = len(X_temp) - int(test_size * len(X_temp))
        X_train = np.append(X_train, X_temp[:train_idx])
        X_test = np.append(X_test, X_temp[train_idx:])
        y_train = np.append(y_train, y_temp[:train_idx])
        y_test = np.append(y_test, y_temp[train_idx:])
    return X_train, X_test, y_train, y_test


