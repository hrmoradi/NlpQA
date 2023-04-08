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
import inspect
import sys

from datetime import datetime

from transformers import AutoModel, TFAutoModel, TFDistilBertPreTrainedModel
from transformers import TFDistilBertForQuestionAnswering, TFBertForQuestionAnswering,  TFGPTJForQuestionAnswering, TFAutoModelForQuestionAnswering
from transformers import AutoTokenizer, AutoConfig
from transformers.modeling_tf_utils import TFQuestionAnsweringLoss, unpack_inputs, get_initializer, input_processing, TFPreTrainedModel
from transformers.models.xlnet.modeling_tf_xlnet import TFXLNetForQuestionAnsweringSimpleOutput, TFXLNetPreTrainedModel
from transformers.modeling_tf_outputs import TFQuestionAnsweringModelOutput

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from keras import backend as K
import torch
import sentencepiece
seed = 99
# os.environ['PYTHONHASHSEED'] = str(seed)
# os.environ['CUDA_VISIBLE_DEVICES'] = ''
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)