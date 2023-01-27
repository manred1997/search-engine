import argparse
import logging
import os
import sys

AUTO_COMPLETE_PATH = os.environ.get("AUTO_COMPLETE_PATH")
sys.path.append(AUTO_COMPLETE_PATH)

from src.ranking.mpc import get_frequence_of_query
from src.retrieval.binarysearch import bisect_contains_check, bisect_list_slice
from src.utils.metric import mean_reciprocal_rank
from src.utils.utils import _read_text_file, init_logger
from tqdm import tqdm

logger = logging.getLogger(__name__)


# Most Popular Completions
def mpc(
    prefix_queries: list, pids: list, query_logs: list, qids: list, topK: int
) -> float:

    sumRR = 0

    # Prepare data
    tf_queries, tfmc_queries = get_frequence_of_query(query_logs)
    sorted_queries = sorted([q[0] for q in tfmc_queries])

    for pr_q, pid in tqdm(zip(prefix_queries, pids)):
        qid = pid.split(".")[1]
        in_u = query_logs[qids.index(qid)]
        if bisect_contains_check(sorted_queries, pr_q):
            # Retrival phase
            try:
                retrive_candidate = bisect_list_slice(sorted_queries, pr_q)
            except Exception:
                continue
            # ReRank phase
            rerank_candidate = []
            for candidate in retrive_candidate:
                rerank_candidate.append([candidate, tf_queries.get(candidate, 0)])
            rerank_candidate.sort(key=lambda x: str(x[1]), reverse=True)
            rerank_candidate = [i[0] for i in rerank_candidate[:topK]]

            sumRR += mean_reciprocal_rank(rerank_candidate, [in_u])

    MRR = sumRR / len(prefix_queries)
    logger.info(f"Mean Reciprocal Rank: {MRR:4f}")
    return MRR


def main(args):
    init_logger()
    # Preprocessing
    prefix_queries = _read_text_file(
        os.path.join(args.source_dir, args.mode, args.prefix)
    )
    pids = _read_text_file(os.path.join(args.source_dir, args.mode, args.pid))
    assert len(prefix_queries) == len(pids)

    queries = _read_text_file(os.path.join(args.source_dir, args.mode, args.query))
    qids = _read_text_file(os.path.join(args.source_dir, args.mode, args.qid))
    assert len(queries) == len(qids)

    # Evaluate
    mpc(
        prefix_queries=prefix_queries,
        pids=pids,
        query_logs=queries,
        qids=qids,
        topK=args.topK,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=str, default="pid.txt")
    parser.add_argument("--prefix", type=str, default="prefix.txt")
    parser.add_argument("--qid", type=str, default="qid.txt")
    parser.add_argument("--query", type=str, default="query.txt")
    parser.add_argument("--time", type=str, default="time.txt")
    parser.add_argument("--uid", type=str, default="uid.txt")
    parser.add_argument("--mode", type=str, default="test")

    # Path file
    parser.add_argument(
        "--source_dir", default=None, required=True, type=str, help="Path to load data"
    )
    parser.add_argument(
        "--target_dir",
        default="../data",
        required=True,
        type=str,
        help="Path to save result",
    )

    parser.add_argument("--topK", type=int, default=4)

    args = parser.parse_args()

    main(args)
