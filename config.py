###################################################################################
################################## DATA RELATED ###################################
###################################################################################
VOCABULARY = {
    "PAD": 0,
    "<s>": 1,
    "<e>": 2,
    "A": 3,
    "B": 4,
    "C": 5
}
INVERSE_VOCABULARY = {v: k for k, v in VOCABULARY.items()}
SPECIAL_TOKENS = ["PAD", "<s>", "<e>"]
NORMAL_TOKENS = [token for token in VOCABULARY if token not in SPECIAL_TOKENS]
SENTENCE_LEN = 16


###################################################################################
################################## CONFIGURATION ##################################
###################################################################################
class TransformerConfig:
    block_size: int = SENTENCE_LEN
    vocab_size: int = len(VOCABULARY)
    n_encoder_layer: int = 6
    n_decoder_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

TRANSFORMER_CONFIG = TransformerConfig
