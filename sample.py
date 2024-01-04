"""
Sample from a trained model.

Given a sentence of characters in the vocabulary, the model will predict the
sorted version of the sentence.

Usage:
$ python sample.py --model_path <path_to_model> --sentence <sentence_to_process>

Example:
$ python sample.py --model_path last_model.pt --sentence ABCABB
"""
import argparse
import torch

from model import Transformer
from config import TRANSFORMER_CONFIG, SENTENCE_LEN
from utils import detokenize_sentence

# Create ArgumentParser object
parser = argparse.ArgumentParser(description='Process a sentence.')
# Add argument for the model path
parser.add_argument('--model_path', type=str, help='The path to the model.')
# Add argument for the sentence
parser.add_argument('--sentence', type=str, help='The sentence to process.')
# Parse the command-line arguments
args = parser.parse_args()
# Convert the sentence to a list of characters
characters = list(args.sentence)

# Determine the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the checkpoint
checkpoint = torch.load(args.model_path, map_location=device)
checkpoint_inverse_vocabulary = checkpoint["config"]["inverse_vocabulary"]

# Check that the sentence is in the vocabulary
for char in characters:
    assert char in checkpoint_inverse_vocabulary.values(), \
        f"The character {char} is not in the trained vocabulary"

# Load the model
model = Transformer(TRANSFORMER_CONFIG)
model.load_state_dict(checkpoint["model"])
model.to(device)

# Prepare the sentence for the model => delimiters and padding
src = ["<s>"] + characters + ["<e>"] 
padd_needed = SENTENCE_LEN - len(src)
src += ["PAD"] * padd_needed

out_tokens = model.generate(src, device=device, top_k=1)[0].tolist()
out_str = detokenize_sentence(out_tokens, checkpoint_inverse_vocabulary)
# Join and remove the delimiters and padding from the output
out_str = "".join(out_str)
out_str = out_str.replace("<s>", "").replace("<e>", "").replace("PAD", "")
# Check correctness
is_correct = "".join(sorted(characters)) == out_str
# Print the results
print(f"Input: {args.sentence}")
print(f"Output: {out_str} {'(correct)' if is_correct else '(incorrect)'}")
