import pandas as pd
from config import VOCABULARY, SENTENCE_LEN
from utils import generate_sentence_pairs

VAL_SIZE = 5000
TEST_SIZE = 1000

# Generate the validation set, check that there are no duplicates
val_set = {"src": [], "tgt_shifted": [], "tgt": []}
while len(val_set["src"]) < VAL_SIZE:
    src, tgt_shifted, tgt = generate_sentence_pairs(VOCABULARY, SENTENCE_LEN)
    if src not in val_set["src"]:
        val_set["src"].append(src)
        val_set["tgt_shifted"].append(tgt_shifted)
        val_set["tgt"].append(tgt)


# Generate the test set, check that there are no duplicates and overlaps with the validation set
test_set = {"src": [], "tgt_shifted": [], "tgt": []}
while len(test_set["src"]) < TEST_SIZE:
    src, tgt_shifted, tgt = generate_sentence_pairs(VOCABULARY, SENTENCE_LEN)
    if src not in val_set["src"] and src not in test_set["src"]:
        test_set["src"].append(src)
        test_set["tgt_shifted"].append(tgt_shifted)
        test_set["tgt"].append(tgt)


# Save the validation and test sets
pd.DataFrame(val_set).to_csv("val.csv", index=False)
pd.DataFrame(test_set).to_csv("test.csv", index=False)

print("Data generated successfully!")
