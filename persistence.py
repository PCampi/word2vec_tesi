"""Persistence module."""
import pickle


def triplet(k, tpl):
    w, s = tpl
    return k, w, s


def dict_to_triplets(d):
    result = [triplet(k, tpl) for k in d for tpl in d[k]]
    return result


def save_csv(set_of_triplets, file_name):
    """Save the set of triplets to a csv file."""
    with open(file_name, "w") as f:
        for emotion, word, score in set_of_triplets:
            line = "{},{},{:.3f}\n".format(emotion, word, score)
            f.write(line)


def save_pickle(obj, file_name):
    """Save a pickle file."""
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f)


def save_results(dict_result, file_name="result.csv"):
    """Save the results to a file."""
    fname, extension = file_name.split('.')

    if extension == "csv":
        triplets = dict_to_triplets(dict_result)
        save_csv(triplets, file_name)
    elif extension == "pickle":
        save_pickle(dict_result, file_name)

    print("Done saving results to file {}".format(file_name))
