import pandas as pd
import numpy as np
import transformers
import os
from tqdm.auto import tqdm  # for showing progress bar
import datasets
import collections
import evaluate
import random
import platform
import re
import copy

from datetime import datetime
from transformers import pipeline
from transformers import DistilBertTokenizer, TFDistilBertForQuestionAnswering, TFBertForQuestionAnswering,  TFGPTJForQuestionAnswering
from transformers import AutoTokenizer
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
import torch

seed = 99
# os.environ['PYTHONHASHSEED'] = str(seed)
# os.environ['CUDA_VISIBLE_DEVICES'] = ''
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)