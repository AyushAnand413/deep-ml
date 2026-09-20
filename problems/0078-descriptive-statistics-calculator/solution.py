import numpy as np

def descriptive_statistics(data: list | np.ndarray) -> dict:
    """
    Calculate various descriptive statistics metrics for a given dataset.
    
    Args:
        data: List or numpy array of numerical values
    
    Returns:
        Dictionary containing mean, median, mode, variance, standard deviation,
        percentiles (25th, 50th, 75th), and interquartile range (IQR)
    """

    # Mean
    mean = np.mean(data)

    # Median
    median = np.median(data)

    # Mode
    values, counts = np.unique(data, return_counts=True)
    mode = values[np.argmax(counts)]

    # Variance
    var = np.var(data)

    # Standard deviation
    standard_deviation = np.std(data)

    # Percentiles
    percentile_25 = np.percentile(data, 25)
    percentile_50 = np.percentile(data, 50)
    percentile_75 = np.percentile(data, 75)

    # Interquartile range
    iqr = percentile_75 - percentile_25

    return {
        "mean": float(mean),
        "median": float(median),
        "mode": int(mode),
        "variance": float(var),
        "standard_deviation": float(standard_deviation),
        "25th_percentile": float(percentile_25),
        "50th_percentile": float(percentile_50),
        "75th_percentile": float(percentile_75),
        "interquartile_range": float(iqr)
    }