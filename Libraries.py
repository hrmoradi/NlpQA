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
import accelerate

from datetime import datetime
from math import ceil

from transformers import AutoModel, TFAutoModel, TFDistilBertPreTrainedModel, AdamW, get_scheduler
from transformers import TFDistilBertForQuestionAnswering, TFBertForQuestionAnswering,  TFGPTJForQuestionAnswering, TFAutoModelForQuestionAnswering
from transformers import AutoTokenizer, AutoConfig, LlamaTokenizer, LlamaConfig, GPT2TokenizerFast

from transformers.modeling_tf_utils import TFPreTrainedModel, TFQuestionAnsweringLoss, unpack_inputs, get_initializer
from transformers.models.xlnet.modeling_tf_xlnet import TFXLNetForQuestionAnsweringSimpleOutput, TFXLNetPreTrainedModel
from transformers.modeling_tf_outputs import TFQuestionAnsweringModelOutput

from transformers.modeling_outputs import QuestionAnsweringModelOutput
from transformers.modeling_utils import PreTrainedModel

from modeling_tf_llama import TFLlamaModel
from transformers import TrainingArguments, Trainer, DefaultDataCollator

from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from keras import backend as K

import torch
from torch import nn
from torch.utils.data import DataLoader

import sentencepiece
seed = 99
# os.environ['PYTHONHASHSEED'] = str(seed)
# os.environ['CUDA_VISIBLE_DEVICES'] = ''
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)