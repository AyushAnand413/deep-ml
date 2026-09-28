import numpy as np
from collections import Counter

def dummy_classifier(y_train, n_test, strategy, constant=None):
    """
    Produce baseline predictions of length n_test using the given strategy.

    Returns a Python list of predicted labels.
    """

    classes = sorted(set(y_train))

    if strategy == "most_frequent":
        counts = Counter(y_train)
        best = max(classes, key=lambda c: (counts[c], -c))
        return [best] * n_test

    elif strategy == "constant":
        return [constant] * n_test

    elif strategy == "uniform":
        return [classes[i % len(classes)] for i in range(n_test)]

    elif strategy == "stratified":
        counts = Counter(y_train)
        n = len(y_train)

        expected = {
            c: n_test * counts[c] / n
            for c in classes
        }

        allocation = {
            c: int(expected[c])
            for c in classes
        }

        remaining = n_test - sum(allocation.values())

        order = sorted(
            classes,
            key=lambda c: (
                -(expected[c] - allocation[c]),
                c
            )
        )

        for c in order[:remaining]:
            allocation[c] += 1

        result = []

        for c in classes:
            result.extend([c] * allocation[c])

        return result

    else:
        raise ValueError("Unknown strategy")