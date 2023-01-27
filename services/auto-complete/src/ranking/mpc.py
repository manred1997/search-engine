from collections import Counter


def get_frequence_of_query(queries: list):
    tf_queries = Counter(queries)
    tfmc_queries = tf_queries.most_common()
    return tf_queries, tfmc_queries
