from matplotlib import pyplot as plt
import numpy as np


def running_average(X, window_size=5):
    """
    Calculates the moving average for each row of a 2D array.
    Each row is transformed into a series of moving averages over the preceding values.

    Args:
        X (np.ndarray): 2D array of shape (n_samples, n_features).
        window_size (int): Size of the moving average window.

    Returns:
        np.ndarray: 2D array of shape (n_samples, n_features - window_size + 1)
                    containing the calculated moving average for each row.
    """
    if window_size < 1:
        raise ValueError("window_size must be at least 1")

    n_samples, n_features = X.shape
    new_length = n_features - window_size + 1
    if new_length < 1:
        raise ValueError("window_size is too large compared to the number of elements per row")

    result = np.empty((n_samples, new_length))
    for i in range(n_samples):
        result[i] = np.convolve(X[i], np.ones(window_size)/window_size, mode='valid')
    return result


def print_beats(X, y):
    unique_beats = np.unique(y)
    n = len(unique_beats)
    cols = 2
    rows = (n + 1) // 2  

    fig, axes = plt.subplots(rows, cols, figsize=(10, 2.5 * rows))
    axes = axes.flatten() 

    for i, beat_type in enumerate(unique_beats):
        ax = axes[i]
        ax.axis(False)
        derired_index = 2

        beat_elements = X[y == beat_type]
        idx = derired_index if derired_index < len(beat_elements) else len(beat_elements) - 1
        ax.plot(beat_elements[idx]) 
        ax.set_title(f"Battito {beat_type}")
        ax.axis('off') 
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()

def print_histogram(y):
    categories, counts = np.unique(y, return_counts=True)

    # Creazione dell'istogramma
    plt.bar(categories, counts, color='skyblue', edgecolor='black')
    plt.xlabel('Categorie')
    plt.ylabel('Numero')
    plt.title('Istogramma delle categorie')
    plt.grid(axis='y', linestyle='--', alpha=0.7)