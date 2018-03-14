# coding:utf-8
from __future__ import unicode_literals, print_function, division
from io import open
import math
import time
import re
import random
import config
import jieba

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()

entPattern = re.compile('<.*>')
yearPattern = re.compile('\d+年')
monthPattern = re.compile('\d+月')
dayPattern = re.compile('\d+[日|号]')

SOS = 0
EOS = 1

toyDataPath = "toypairs"
synDataPath = "coreqa_data/syn_data/qa_pairs"
cqaDataPath = "coreqa_data/cqa_data/cqa_triple_origina_v0"
