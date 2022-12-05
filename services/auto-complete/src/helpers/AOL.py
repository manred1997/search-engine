"""
Split AOL query log dataset into {train, valid, test}.{query, uid, time}.txt
"""
import datetime

import os
import logging
from tqdm import tqdm

from utils.utils import normalize_string

logger = logging.getLogger(__name__)

def process(args,
            splits = ['train', 'valid', 'test'],
            columns = ['uid', 'query', 'time'],
            fmt = '%Y-%m-%d %H:%M:%S'):
    
    itv = {s: tuple(vars(args)[f"{s}_{i}"] for i in ['start', 'end']) for s in splits}
    for s in splits:
        logger.info(f"  {s:5s} data: from {itv[s][0]} until {itv[s][1]}")

    itv = {k: tuple(datetime.datetime.strptime(x, fmt) for x in v) for k, v in itv.items()}

    valid = (itv['train'][0] < itv['train'][1] <= itv['valid'][0] < itv['valid'][1] <= itv['test'][0] < itv['test'][1])
    assert valid, "Invalid time intervals"

    # make directory and open files to write
    target_dir = args.target_dir
    os.makedirs(target_dir, exist_ok=True)
    f = {s: {column: open(os.path.join(target_dir, f"{s}.{column}.txt"), 'w') for column in columns} for s in splits}

    # read original AOL query log dataset and write data into files
    cnt = {s: 0 for s in splits}
    for i in range(1, 11):
        filename = f"user-ct-test-collection-{i:02d}.txt"
        logger.info(f"Reading {filename}...")
        f_org = open(os.path.join(args.aol_benchmark_dir, filename))
        f_org.readline() # Skip first row
        prev = {column: '' for column in columns}
        for line in tqdm(f_org):
            data = {column: v for column, v in zip(columns, line.strip().split('\t')[:3])}
            # normalize queries
            data['query'] = normalize_string(data['query'])
            # filter out too short queries and redundant queries
            # data['query'] == '-'
            if len(data['query']) < 3 or (data['uid'], data['query']) == (prev['uid'], prev['query']):
                continue
            t = datetime.datetime.strptime(data['time'], fmt)
            for s in splits:
                if itv[s][0] <= t < itv[s][1]:
                    cnt[s] += 1
                    for column in columns:
                        f[s][column].write(data[column] + '\n')
            prev = data

    for s in splits:
        logger.info(f"Number of {s:5s} data: {cnt[s]:8d}")
