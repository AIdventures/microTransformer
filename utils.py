import random
from config import SPECIAL_TOKENS

###################################################################################
################################## DATA GENERATION ################################
###################################################################################
def calculate_total_possibilities(characters, max_sequence_len):
    """
    Calculate the total number of possibilities for a given characters and
    maximum sentence length
    """
    total_possibilities = 0

    for sequence_len in range(2, max_sequence_len + 1):
        total_possibilities += len(characters) ** sequence_len

    return total_possibilities


def generate_sentence_pairs(vocabulary, sentence_len):
    # Ensure that special tokens are in the vocabulary
    for token in SPECIAL_TOKENS:
        assert token in vocabulary, f"{token} not in vocabulary"
    
    # We know that we need to reserve 2 positions for "<s>" and "<e>"
    # The "real" sentence length is AT MOST sentence_len - 2
    # the remaining positions are filled with "PAD"
    real_sentence_len = random.randint(2, sentence_len - 2)
    pad_token_num = sentence_len - real_sentence_len - 2
    # Now I want to generate a sentence of length real_sentence_len
    # normal_tokens is the vocabulary without "<s>" and "<e>" and "PAD"
    normal_tokens = [token for token in vocabulary if token not in SPECIAL_TOKENS]
    # Generate a sentence of length real_sentence_len
    messy_sentence = random.choices(normal_tokens, k=real_sentence_len)
    shorted_sentence = sorted(messy_sentence)

    src = ["<s>"] + messy_sentence + ["<e>"] + ["PAD"] * pad_token_num
    tgt_shifted = ["<s>"] + shorted_sentence + ["<e>"] + ["PAD"] * pad_token_num
    tgt = shorted_sentence + ["<e>"] + ["PAD"] * (pad_token_num + 1)
    return src, tgt_shifted, tgt


def tokenize_sentence(sentence, vocabulary):
    """
    Tokenize a sentence using the vocabulary
    """
    return [vocabulary[token] for token in sentence]


def detokenize_sentence(sentence, ivocabulary):
    """
    Detokenize a sentence using the inverse vocabulary
    """
    return [ivocabulary[token] for token in sentence]