# -*- coding: utf-8 -*-
"""Multi_Pass_Correction_Pipeline.ipynb

This notebook provides a function to process a multi-sentence text input
using a two-stage correction process with a final disfluency check.
"""

# @title 1. Install Libraries
# This will install all necessary libraries.
!pip install torch transformers huggingface_hub -q

# @title 2. Import Libraries and Set Up Environment
import torch
from transformers import (
    T5ForConditionalGeneration, T5Tokenizer, AutoModelForSeq2SeqLM, AutoTokenizer
)
import re

# --- IMPORTANT CONFIGURATION ---
# Your Hugging Face Hub username
HF_USERNAME = "vamshi0310"

# Repository IDs for your fine-tuned models on the Hub.
DC_REPO_ID = f"{HF_USERNAME}/fine-tuned-disfluency-model"
GEC_MODEL_ID = 'vennify/t5-base-grammar-correction'
# Your Hugging Face token with 'read' permissions.
HF_TOKEN = "YOUR_HF_TOKEN"
# -------------------------------

# Set device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# @title 3. Load Models from Hugging Face Hub
try:
    print("Loading Disfluency Correction model from the Hub...")
    dc_tokenizer = AutoTokenizer.from_pretrained(DC_REPO_ID, token=HF_TOKEN)
    dc_model = T5ForConditionalGeneration.from_pretrained(DC_REPO_ID, token=HF_TOKEN).to(device)
    dc_model.eval()

    print("Loading Grammar Correction model from the Hub...")
    gec_tokenizer = AutoTokenizer.from_pretrained(GEC_MODEL_ID, token=HF_TOKEN)
    gec_model = AutoModelForSeq2SeqLM.from_pretrained(GEC_MODEL_ID, token=HF_TOKEN).to(device)
    gec_model.eval()

    print("\nModels loaded successfully from Hugging Face Hub!")

except Exception as e:
    print(f"Error: Failed to load models from the Hub. Please check your repo IDs and internet connection.")
    print(f"Details: {e}")
    # Stop execution if models fail to load
    raise


# @title 4. Define Correction Functions
# These functions encapsulate the inference logic for each model.

def correct_disfluency(disfluent_sentence, model, tokenizer, device, max_length=128):
    """Corrects a single disfluent sentence using the fine-tuned T5 model."""
    input_text = f"correct disfluency: {disfluent_sentence}"
    inputs = tokenizer([input_text], max_length=max_length, padding='max_length', truncation=True, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, max_length=max_length, num_beams=5, early_stopping=True)

    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


def correct_grammar(incorrect_sentence, model, tokenizer, device, max_length=128):
    """Corrects a single sentence for grammar using the pre-trained GEC model."""
    input_text = f"grammar: {incorrect_sentence}"
    inputs = tokenizer([input_text], max_length=max_length, padding='max_length', truncation=True, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, max_length=max_length, num_beams=5, early_stopping=True)

    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


# @title 5. Define a new function to correct an entire block of text with multiple passes
def correct_text_block_multi_pass(text_block, dc_model, dc_tokenizer, gec_model, gec_tokenizer, device):
    """
    Corrects a block of text using a multi-pass process:
    1. Sentence-by-sentence Disfluency Correction (DC).
    2. One-pass Grammar Correction (GEC) on the entire paragraph.
    3. Final Disfluency Correction Pass (DC) to catch any remaining issues.

    Args:
        text_block (str): The full paragraph of text to correct.

    Returns:
        str: The fully corrected and reconstructed paragraph.
    """
    # Use a simple regex to split the text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text_block)

    dc_corrected_sentences = []

    print("\n--- Step 1: Initial Disfluency Correction (Sentence-by-Sentence) ---")
    for sentence in sentences:
        if not sentence.strip():
            continue

        # Apply Disfluency Correction
        dc_corrected = correct_disfluency(sentence, dc_model, dc_tokenizer, device)
        dc_corrected_sentences.append(dc_corrected)

        print(f"Original: '{sentence}' -> DC Output: '{dc_corrected}'")

    # Combine the DC-corrected sentences into a single block of text for GEC
    combined_dc_text = " ".join(dc_corrected_sentences)

    print("\n--- Step 2: Grammar Correction (Single Pass) ---")
    print(f"Input to GEC: '{combined_dc_text}'")

    # Apply Grammar Correction once to the entire block of text
    gec_output = correct_grammar(combined_dc_text, gec_model, gec_tokenizer, device)

    print(f"GEC Output: '{gec_output}'")

    print("\n--- Step 3: Final Disfluency Pass (Post-GEC) ---")
    # The final pass splits the GEC output back into sentences to check for missed disfluencies
    gec_output_sentences = re.split(r'(?<=[.!?])\s+', gec_output)

    final_sentences = []
    for sentence in gec_output_sentences:
        if not sentence.strip():
            continue

        # Apply Disfluency Correction one last time
        final_corrected = correct_disfluency(sentence, dc_model, dc_tokenizer, device)
        final_sentences.append(final_corrected)

        print(f"Input to final DC: '{sentence}' -> Final Output: '{final_corrected}'")

    # Join the final corrected sentences back into a single paragraph
    return " ".join(final_sentences)


# @title 6. Run the correction on your example text
if dc_model and gec_model:
    your_example = """um so i think we, uh, should, like, really focus on the first part of the presentation and then, uh, move on to the second part quickly. I I believe that is the, you know, right way to go about it."""
    print("\n--- Correcting the full example text block ---")
    final_output = correct_text_block_multi_pass(your_example, dc_model, dc_tokenizer, gec_model, gec_tokenizer, device)

    print("\n" + "="*50)
    print("Original Text:")
    print(your_example)
    print("\nFinal Corrected Text:")
    print(final_output)
    print("="*50)

else:
    print("\nOne or more models failed to load. Cannot run the pipeline.")
