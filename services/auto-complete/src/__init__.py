import logging
import os
import sys

logging.getLogger(__name__).addHandler(logging.StreamHandler())
logging.getLogger(__name__).setLevel(logging.INFO)

sys.path.append(os.getcwd())
