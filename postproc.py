"""Generic postprocessing module."""


def filter_duplicates(arr_of_tuples):
    """Keep only the first occurrence of an element in an array of tuples."""
    uniques = set()
    result = []

    for word, score in arr_of_tuples:
        if word not in uniques:
            uniques.add(word)
            result.append((word, score))

    return result
