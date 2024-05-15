import ast
import pandas as pd
import torch
from torch.utils.data import Dataset
from config import VOCABULARY, NORMAL_TOKENS, SENTENCE_LEN
from utils import (
    calculate_total_possibilities,
    generate_sentence_pairs,
    tokenize_sentence,
)

###################################################################################
################################## CONFIGURATION ##################################
###################################################################################
EVAL_DFS = {"val": pd.read_csv("val.csv"), "test": pd.read_csv("test.csv")}
# Take an example and check sentence length
example = ast.literal_eval(EVAL_DFS["val"].iloc[0]["src"])
assert len(example) == SENTENCE_LEN, "The sentence length is not correct (val)"
example = ast.literal_eval(EVAL_DFS["test"].iloc[0]["src"])
assert len(example) == SENTENCE_LEN, "The sentence length is not correct (test)"


###################################################################################
################################# DATA LOADERS ####################################
###################################################################################
TOTAL_POSSIBILITIES = calculate_total_possibilities(NORMAL_TOKENS, SENTENCE_LEN)
VAL_POSSIBILITIES = len(EVAL_DFS["val"])
TEST_POSSIBILITIES = len(EVAL_DFS["test"])
TRAIN_POSSIBILITIES = TOTAL_POSSIBILITIES - VAL_POSSIBILITIES - TEST_POSSIBILITIES
NUM_POSSIBILITIES = {
    "train": TRAIN_POSSIBILITIES,
    "val": VAL_POSSIBILITIES,
    "test": TEST_POSSIBILITIES,
}

VAL_SRCS = [ast.literal_eval(v) for v in EVAL_DFS["val"]["src"].values]
TEST_SRCS = [ast.literal_eval(v) for v in EVAL_DFS["test"]["src"].values]


class CharactersDataset(Dataset):
    def __init__(self, split):
        """
        Dataset for the characters dataset.
        The validation and test sets are pre-generated at val.csv and test.csv
        The training set is generated on the fly and must don't have overlap
        with the validation and test sets

        Args:
            split (str): "train", "val" or "test"
        """
        self.split = split
        self.num_samples = NUM_POSSIBILITIES[split]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.split == "train":
            src, tgt_shifted, tgt = self.generate_train_sample()
        else:
            src, tgt_shifted, tgt = self.load_eval_sample(idx)

        t_src = tokenize_sentence(src, VOCABULARY)
        t_tgt_shifted = tokenize_sentence(tgt_shifted, VOCABULARY)
        t_tgt = tokenize_sentence(tgt, VOCABULARY)

        return {
            "src": torch.tensor(t_src),
            "tgt_shifted": torch.tensor(t_tgt_shifted),
            "tgt": torch.tensor(t_tgt),
        }

    def generate_train_sample(self):
        """Generate a training sample"""
        is_generated = False
        while not is_generated:
            src, tgt_shifted, tgt = generate_sentence_pairs(VOCABULARY, SENTENCE_LEN)
            if src not in VAL_SRCS and src not in TEST_SRCS:
                is_generated = True

        return src, tgt_shifted, tgt

    def load_eval_sample(self, idx):
        """Load an evaluation sample"""
        src, tgt_shifted, tgt = EVAL_DFS[self.split].iloc[idx].values
        src = ast.literal_eval(src)
        tgt_shifted = ast.literal_eval(tgt_shifted)
        tgt = ast.literal_eval(tgt)
        return src, tgt_shifted, tgt
