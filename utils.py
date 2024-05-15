import torch
import random
from config import SPECIAL_TOKENS


###################################################################################
################################## DATA GENERATION ################################
###################################################################################
def calculate_total_possibilities(characters: list[str], max_sequence_len: int) -> int:
    """
    Calculate the total number of possibilities for a given characters and
    maximum sentence length

    Args:
        characters (list[str]): The characters to consider
        max_sequence_len (int): The maximum sequence length

    Returns:
        int: The total number of possibilities
    """
    total_possibilities = 0

    for sequence_len in range(2, max_sequence_len + 1):
        total_possibilities += len(characters) ** sequence_len

    return total_possibilities


def generate_sentence_pairs(vocabulary: dict[str, int], sentence_len: int):
    """Generate a sentence, the sorted version of the sentence and the target

    Args:
        vocabulary (dict): The vocabulary
        sentence_len (int): The sentence length

    Returns:
        tuple: The source sentence, the sorted sentence and the target sentence
    """
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


def detokenize_sentence(
    sentence: torch.Tensor, ivocabulary: dict[int, str]
) -> list[str]:
    """Detokenize a sentence using the inverse vocabulary

    Args:
        sentence (torch.Tensor): The sentence to detokenize
        ivocabulary (dict): The inverse vocabulary

    Returns:
        list[str]: The detokenized sentence
    """
    return [ivocabulary[token] for token in sentence]
