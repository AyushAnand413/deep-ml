import numpy as np
def empirical_pmf(samples):
    """
    Given an iterable of integer samples, return a list of (value, probability)
    pairs sorted by value ascending.
    """
    # TODO: Implement the function
    #cnt the unqiue values 
    if len(samples)==0:
        return []
    values,count = np.unique(samples,return_counts=True)
    total = len(samples)

    probab = count/total

    return [(int(value), float(probability))
            for value, probability in zip(values, probab)]

