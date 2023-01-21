import os
import logging

import torch

from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class Trainer(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def train(self,  **kwargs):
        pass

    @abstractmethod
    def evaluate(self, **kwargs):
        pass

    def save_model(self):
        pass
    def load_model(self):
        pass

    