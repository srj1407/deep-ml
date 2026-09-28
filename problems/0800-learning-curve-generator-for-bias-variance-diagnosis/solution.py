import numpy as np

def learning_curve(X_train, y_train, X_val, y_val, train_sizes, degree, bias_threshold=0.5, variance_threshold=0.5):
    """
    Generate a learning curve and diagnose bias vs variance.

    Returns a dict with 'train_errors', 'val_errors', and 'diagnosis'.
    """
    train_matrix = []
    val_matrix = []
    train_errors = []
    val_errors = []
    for i in range(len(X_train)):
        temp = []
        for j in range(degree + 1):
            temp.append(np.power(X_train[i][0], j))
        train_matrix.append(temp)
    for i in range(len(X_val)):
        temp = []
        for j in range(degree + 1):
            temp.append(np.power(X_val[i][0], j))
        val_matrix.append(temp)
    train_matrix = np.array(train_matrix)
    val_matrix = np.array(val_matrix)
    for n in train_sizes:
        X_temp = train_matrix[:n]
        y_temp = y_train[:n]
        X_pinv = np.linalg.pinv(X_temp)
        weights = X_pinv @ y_temp
        train_preds = X_temp @ weights
        train_mse = np.mean((y_temp - train_preds) ** 2)
        train_errors.append(train_mse)
        val_preds = val_matrix @ weights
        val_mse = np.mean((y_val - val_preds) ** 2)
        val_errors.append(val_mse)
    final_train_error = train_errors[-1]
    final_val_error = val_errors[-1]
    diagnosis = ''
    if final_train_error > bias_threshold:
        diagnosis = 'high_bias'
    elif (final_val_error - final_train_error) > variance_threshold:
        diagnosis = 'high_variance'
    else:
        diagnosis = 'good_fit'
    return {
        'train_errors': train_errors, 
        'val_errors': val_errors, 
        'diagnosis': diagnosis
    }

