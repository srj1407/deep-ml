import numpy as np
from typing import List, Tuple

def k_fold_cross_validation(n_samples: int, k: int = 5, shuffle: bool = True) -> List[Tuple[List[int], List[int]]]:
    """
    Generate train/test index splits for k-fold cross-validation.
    
    Args:
        n_samples: Total number of samples in the dataset
        k: Number of folds (default 5)
        shuffle: Whether to shuffle indices before splitting (default True)
    
    Returns:
        List of (train_indices, test_indices) tuples
    """
    # Your code here
    res = []
    i = 0
    j = 0
    indices = [num for num in range(n_samples)]
    if shuffle:
        np.random.shuffle(indices)
    for fold in range(k):
        if k == 0:
            j = int(n_samples / k) + (n_samples % k)
        else:
            j = int(n_samples / k)
        test_indices = indices[i:i+j]
        train_indices = [x for x in indices if x not in test_indices]
        item = [train_indices, test_indices]
        item = tuple(item)
        res.append(item)
        i = i+j
    return res
