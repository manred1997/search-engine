"""
Split AOL query log dataset into {train, valid, test}.{query, uid, time}.txt
"""
import datetime

import os
import logging
from tqdm import tqdm

from src.utils.normalize import (
    normalize_diacritic,
    normalize_encode
)

from src.helpers.synthetic import (
    prefixSyntheszie
)

logger = logging.getLogger(__name__)

def process(train_start = '2006-03-01 00:00:00',
            train_end = '2006-05-18 00:00:00',
            test_start = '2006-05-25 00:00:00',
            test_end = '2006-06-01 00:00:00',
            dev_start = '2006-05-18 00:00:00',
            dev_end = '2006-05-25 00:00:00',
            splits = ['train', 'test', 'dev'],
            columns = ['uid', 'query', 'time'],
            fmt = '%Y-%m-%d %H:%M:%S',
            source_dir = './aol_benchmark_dir',
            target_dir = './data') -> None:
    
    # PWD = os.environ.get('PWD')
    # source_dir = os.path.join(PWD, source_dir)

    itv = {}
    if 'train' in splits and train_start and train_end:
        itv['train'] = tuple((train_start, train_end))
    
    if 'test' in splits and test_start and test_end:
        itv['test'] = tuple((test_start, test_end))
    
    if 'dev' in splits and dev_start and dev_end:
        itv['dev'] = tuple((dev_start, dev_end))
    
    for s in splits:
        logger.info(f"  {s:5s} data: from {itv[s][0]} until {itv[s][1]}")

    # normalize time
    itv = {k: tuple(datetime.datetime.strptime(x, fmt) for x in v) for k, v in itv.items()}

    valid = (itv['train'][0] < itv['train'][1] <= itv['dev'][0] < itv['dev'][1] <= itv['test'][0] < itv['test'][1])
    assert valid, "Invalid time intervals"

    # make directory and open files to write
    for s in splits:
        if not os.path.exists(os.path.join(target_dir, s)):
            os.makedirs(os.path.join(target_dir, s))

    f = {s: {column: open(os.path.join(target_dir, f"{s}", f"{column}.txt"), 'w') for column in columns} for s in splits}

    for s in splits:
        for c in ['prefix', 'pid']:
            f[s][c] = open(os.path.join(target_dir, f"{s}", f"{c}.txt"), 'w')

    # read original AOL query log dataset and write data into files
    cnt = {s: 0 for s in splits}
    qid = 1

    for i in range(1, 11):
        filename = f"user-ct-test-collection-{i:02d}.txt"
        logger.info(f"Reading {filename}...")
        f_org = open(os.path.join(source_dir, filename))
        f_org.readline() # Skip first row
        prev = {column: '' for column in columns}
        for line in tqdm(f_org):
            data = {column: v for column, v in zip(columns, line.strip().split('\t')[:3])}
            # normalize queries
            data['query'] = normalize_encode(data['query'])
            data['query'] = normalize_diacritic(data['query'])
            # filter out too short queries and redundant queries
            # data['query'] == '-'
            if len(data['query']) < 3 or len(data['query'].split()) < 2 or (data['uid'], data['query']) == (prev['uid'], prev['query']):
                continue
            t = datetime.datetime.strptime(data['time'], fmt)
            for s in splits:
                if itv[s][0] <= t < itv[s][1]:
                    cnt[s] += 1
                    for column in columns:
                        f[s][column].write(data[column] + '\n')
                    
                    prefixes = prefixSyntheszie(data['query'])
                    pid = 1
                    for prefix in prefixes:
                        f[s]['prefix'].write(prefix + '\n')
                        f[s]['pid'].write(f"{data['uid']}.{qid}.{pid}" + '\n')
                        pid += 1
            prev = data
            qid += 1

    for s in splits:
        logger.info(f"Number of {s:5s} data: {cnt[s]:8d}")
