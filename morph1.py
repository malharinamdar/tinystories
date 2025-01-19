from transformers import AutoTokenizer
import numpy as np
import os
import stanza  # Import stanza for MorphScore calculation
from tqdm import tqdm  # Import tqdm for progress bar

# Initialize Stanza for Hindi
stanza.download('hi')  # Download Hindi model if not already done
nlp = stanza.Pipeline('hi')  # Initialize the pipeline

# Paths to existing binary files
train_bin_path = '/home/user/nanogpt/data/hindisutra/train.bin'
val_bin_path = '/home/user/nanogpt/data/hindisutra/val.bin'

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("TWO/sutra-mlt256-v2")

def calculate_morphscore(text):
    """Calculate Morphological Complexity Score using Stanza."""
    doc = nlp(text)
    
    morph_scores = []
    
    for sentence in doc.sentences:
        for word in sentence.words:
            morph_features = len(word.feats.split('|')) if word.feats else 0
            morph_scores.append(morph_features)

    return np.mean(morph_scores) if morph_scores else 0

def tokens_to_text(tokens):
    """
    Convert token IDs to text using the tokenizer.
    Args:
        tokens (list[int]): List of token IDs.
    Returns:
        str: Decoded text.
    """
    return tokenizer.decode(tokens, skip_special_tokens=True)

def process_bin_file(bin_path):
    """
    Process the binary file to calculate MorphScore.
    Args:
        bin_path (str): Path to the binary file.
    Returns:
        float: Average Morphological Complexity Score.
    """
    # Load binary data
    data = np.memmap(bin_path, dtype=np.uint32, mode='r')
    
    # BOS and EOS tokens
    bos_token = 0  # BOS token ID in the binary file
    eos_token = 2  # EOS token ID in the binary file
    scores = []
    current_index = 0
    
    # Iterate through the binary data to detect stories based on BOS/EOS tokens
    while current_index < len(data):
        # Find the next BOS token
        start_indices = np.where(data[current_index:] == bos_token)[0]
        if len(start_indices) == 0:
            break
        start_idx = start_indices[0] + current_index
        
        # Find the next EOS token after BOS
        end_indices = np.where(data[start_idx + 1:] == eos_token)[0]
        if len(end_indices) == 0:
            break
        end_idx = end_indices[0] + start_idx + 1
        
        # Extract tokens for the current story, excluding BOS and EOS
        story_tokens = data[start_idx + 1:end_idx]
        current_index = end_idx + 1  # Move to the next story
        
        # Convert tokens to text
        story_text = tokens_to_text(story_tokens)

        # Calculate MorphScore for the story
        score = calculate_morphscore(story_text)
        scores.append(score)

        # Update progress bar
        tqdm.write(f"Processed story from index {start_idx} to {end_idx}, Score: {score}")

    return np.mean(scores) if scores else 0




if __name__ == "__main__":
    # Initialize tqdm progress bar
    with tqdm(total=2, desc="Processing Files") as pbar:
        # Calculate MorphScore for train dataset
        avg_train_score = process_bin_file(train_bin_path)
        print(f"Average Morphological Complexity Score (Train): {avg_train_score}")
        pbar.update(1)

        # Calculate MorphScore for validation dataset
        avg_val_score = process_bin_file(val_bin_path)
        print(f"Average Morphological Complexity Score (Validation): {avg_val_score}")
        pbar.update(1)
