import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


def get_evals(
    preds: List[str], sources: List[str], targets: List[str], is_lower=False
) -> Tuple[int, ...]:
    """
    This function evaluates the accuarcy of each token in sentence in batch
    :param preds List[str], sources List[str], targets List[str]
    :return
        no. of correction to correction tokens

        no. of correction to incorrection tokens

        no. of incorrection to correction tokens

        no. of incorrection to incorrection tokens

    """

    correct2correct = 0
    correct2incorrect = 0
    incorrect2correct = 0
    incorrect2incorrect = 0

    is_same_words = None
    if is_lower:
        is_same_words = lambda word_1, word_2: word_1.lower() == word_2.lower()
    else:
        is_same_words = lambda word_1, word_2: word_1 == word_2

    assert len(preds) == len(targets) == len(sources)

    for pred, source, target in zip(preds, sources, targets):

        pred = pred.split()
        source = source.split()
        target = target.split()

        assert len(pred) == len(source) == len(target)

        for p, s, t in zip(pred, source, target):
            if is_same_words(s, t) and is_same_words(p, t):
                correct2correct += 1
            elif is_same_words(s, t) and not is_same_words(p, t):
                correct2incorrect += 1
            elif not is_same_words(s, t) and is_same_words(p, t):
                incorrect2correct += 1
            elif not is_same_words(s, t) and not is_same_words(p, t):
                incorrect2incorrect += 1

    return (correct2correct, correct2incorrect, incorrect2correct, incorrect2incorrect)


def get_word_correction_rate(incorr2corr: int, incorr2incorr: int) -> float:
    return (incorr2corr) / (incorr2corr + incorr2incorr)


def get_total_words(
    corr2corr: int, corr2incorr: int, incorr2corr: int, incorr2incorr: int
) -> int:
    return corr2corr + corr2incorr + incorr2corr + incorr2incorr


def get_word_accuracy(
    corr2corr: int, corr2incorr: int, incorr2corr: int, incorr2incorr: int
) -> float:
    return (corr2corr + incorr2corr) / (
        corr2corr + corr2incorr + incorr2corr + incorr2incorr
    )
