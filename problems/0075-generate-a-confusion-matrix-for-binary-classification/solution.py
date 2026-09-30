from collections import Counter

def confusion_matrix(data):
    # Count each (actual, predicted) pair
    counts = Counter(map(tuple, data))

    # True Positive: actual = 1, predicted = 1
    tp = counts[(1, 1)]

    # False Positive: actual = 0, predicted = 1
    fp = counts[(0, 1)]

    # False Negative: actual = 1, predicted = 0
    fn = counts[(1, 0)]

    # True Negative: actual = 0, predicted = 0
    tn = counts[(0, 0)]
    return [[tp, fn], [fp, tn]]