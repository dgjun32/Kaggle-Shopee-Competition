import transformers
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained(cfg_bert.model_name)
backbone = BertModel.from_pretrained(cfg_bert.model_name)

