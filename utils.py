def list_split(abstracts: list, objects: list):
    """ 根据 objects 拆分 abstracts，并使得左右两部分的长度尽可能接近

    :param abstracts:
    :param objects:
    :return: left part and right part
    """
    candidates = []
    for i in range(len(abstracts) - len(objects)):
        if abstracts[i:i + len(objects)] == objects:
            candidates.append(i)
    if not candidates:
        raise ValueError
    min_diff, best = 99999, 0
    for start in candidates:
        diff = abs(2 * start - len(abstracts) + len(objects))
        if diff < min_diff:
            min_diff = diff
            best = start
    return abstracts[:best], abstracts[best + len(objects):]


def list_gather(inputs: list, idx):
    return [inputs[idx[i]] for i in range(len(idx))]
