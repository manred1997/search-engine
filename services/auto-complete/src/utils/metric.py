

def mean_reciprocal_rank(source: list,  target: list):
    mrr = 0
    for i, s in enumerate(source):
        if s in target:
            mrr = 1/(i+1)
            break
    return mrr
    # mrr = 0
    # index = len(source) + 1
    # for s in source:
    #     index -= 1
    #     if s in target:
    #         mrr = index/len(source)

    return mrr