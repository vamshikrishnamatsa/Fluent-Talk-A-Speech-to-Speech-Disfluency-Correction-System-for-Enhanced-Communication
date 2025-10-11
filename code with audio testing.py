# -- coding: utf-8 --
"""Final_Complete_Pipeline_with_Reward_Model_and_Voice_Input.ipynb

This notebook provides a complete, end-to-end script for the multi-pass correction
pipeline, including voice recording, transcription, feedback generation, and a
fluency score from the Reward Model.
"""

# @title 1. Install Libraries (Updated)
# Added whisper for transcription and librosa for audio processing
!pip install torch transformers huggingface_hub sentencepiece accelerate datasets librosa soundfile -q
# Install ffmpeg for audio processing, required by librosa on some systems
!apt-get install -y -qq ffmpeg

# @title 2. Import Libraries and Set Up Environment (Updated)
import torch
from transformers import (
    T5ForConditionalGeneration, T5Tokenizer, AutoModelForSeq2SeqLM, AutoTokenizer,
    AutoModelForSequenceClassification, WhisperForConditionalGeneration, WhisperProcessor
)
import re
import difflib
import librosa
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode
import numpy as np
from scipy.io.wavfile import read, write

# --- IMPORTANT CONFIGURATION ---
HF_USERNAME = "vamshi0310"
DC_REPO_ID = f"{HF_USERNAME}/fine-tuned-disfluency-model"
REWARD_REPO_ID = f"{HF_USERNAME}/custom-fluency-classifier"
GEC_MODEL_ID = 'vennify/t5-base-grammar-correction'
# Use a robust Speech-to-Text model like Whisper
STT_MODEL_ID = "openai/whisper-base"
# Your Hugging Face token with 'read' permissions.
HF_TOKEN = ""
# -------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# @title 3. Load All Models from Hugging Face Hub (Updated)
try:
    print("Loading Speech-to-Text (Whisper) model from the Hub...")
    stt_processor = WhisperProcessor.from_pretrained(STT_MODEL_ID, token=HF_TOKEN)
    stt_model = WhisperForConditionalGeneration.from_pretrained(STT_MODEL_ID, token=HF_TOKEN).to(device)
    stt_model.eval()

    print("Loading Disfluency Correction model from the Hub...")
    dc_tokenizer = AutoTokenizer.from_pretrained(DC_REPO_ID, token=HF_TOKEN)
    dc_model = T5ForConditionalGeneration.from_pretrained(DC_REPO_ID, token=HF_TOKEN).to(device)
    dc_model.eval()

    print("Loading Grammar Correction model from the Hub...")
    gec_tokenizer = AutoTokenizer.from_pretrained(GEC_MODEL_ID, token=HF_TOKEN)
    gec_model = AutoModelForSeq2SeqLM.from_pretrained(GEC_MODEL_ID, token=HF_TOKEN).to(device)
    gec_model.eval()

    print("Loading Reward Model from the Hub...")
    reward_tokenizer = AutoTokenizer.from_pretrained(REWARD_REPO_ID, token=HF_TOKEN)
    reward_model = AutoModelForSequenceClassification.from_pretrained(REWARD_REPO_ID, token=HF_TOKEN).to(device)
    reward_model.eval()

    print("\nAll models loaded successfully from Hugging Face Hub!")
except Exception as e:
    print(f"Error: Failed to load models from the Hub. Please check your repo IDs and internet connection.")
    print(f"Details: {e}")
    raise

# @title 4. Define Audio Recording and Transcription Functions (Corrected)
def record_audio(filename="audio.webm"):
  """
  Records audio from the browser microphone using a robust, promise-based
  approach that waits for the user to finish.
  """
  js_code = """
    async function recordAudio() {
      // Use a promise to wait for the recording to complete
      const audioPromise = new Promise(resolve => {
        // Create the UI elements
        const div = document.createElement('div');
        const startButton = document.createElement('button');
        startButton.textContent = 'Start Recording';
        div.appendChild(startButton);

        const statusP = document.createElement('p');
        statusP.textContent = 'Click the button to start recording.';
        div.appendChild(statusP);

        const preview = document.createElement('audio');
        preview.controls = true;
        div.appendChild(preview);

        // Append the UI to the output cell
        document.body.appendChild(div);

        let stream;
        let recorder;
        let chunks = [];

        startButton.onclick = async () => {
          if (recorder && recorder.state === 'recording') {
            // --- STOP RECORDING ---
            recorder.stop();
            stream.getTracks().forEach(track => track.stop()); // Stop microphone access
            startButton.disabled = true;
            statusP.textContent = 'Processing audio...';
          } else {
            // --- START RECORDING ---
            try {
              stream = await navigator.mediaDevices.getUserMedia({ audio: true });
              recorder = new MediaRecorder(stream);

              recorder.ondataavailable = e => chunks.push(e.data);

              recorder.onstop = () => {
                const blob = new Blob(chunks, { type: 'audio/webm' });
                const url = URL.createObjectURL(blob);
                preview.src = url;

                const reader = new FileReader();
                reader.readAsDataURL(blob);
                reader.onloadend = () => {
                  // Resolve the promise with the base64 data, sending it to Python
                  resolve(reader.result);
                  div.remove(); // Clean up the UI
                };
              };

              chunks = [];
              recorder.start();
              startButton.textContent = 'Stop Recording';
              statusP.textContent = 'Recording... Click button to stop.';

            } catch (err) {
              console.error('Error getting audio stream:', err);
              statusP.textContent = 'Error: Could not access microphone. Please allow permission and try again.';
            }
          }
        };
      });
      const base64data = await audioPromise;
      return base64data;
    }
    recordAudio();
  """
  try:
    print("A recording interface will appear. Please grant microphone permissions in your browser.")
    # eval_js will execute the JS and wait for the promise to resolve
    base64_data_str = eval_js(js_code)

    # Decode the base64 string (it includes a prefix like "data:audio/webm;base64,")
    data = b64decode(base64_data_str.split(',')[1])

    with open(filename, 'wb') as f:
        f.write(data)

    return filename
  except Exception as e:
    print(f"An error occurred during audio recording: {e}")
    return None

def transcribe_audio(filepath, model, processor, device):
    """Transcribes audio file to text using Whisper."""
    if not filepath:
        return "[Audio recording failed]"
    try:
        # Load audio file and resample to the 16kHz required by Whisper
        audio_array, sampling_rate = librosa.load(filepath, sr=16000)
    except Exception as e:
        print(f"Error loading audio file with librosa: {e}")
        return "[Transcription failed]"

    print("Transcribing audio...")
    input_features = processor(audio_array, sampling_rate=16000, return_tensors="pt").input_features.to(device)

    with torch.no_grad():
        predicted_ids = model.generate(input_features)

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    print("Transcription complete.")
    return transcription

# @title 5. Define Correction and Feedback Functions (Unchanged)
def correct_disfluency(disfluent_sentence, model, tokenizer, device, max_length=128):
    input_text = f"correct disfluency: {disfluent_sentence}"
    inputs = tokenizer([input_text], max_length=max_length, padding='max_length', truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        generated_ids = model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, max_length=max_length, num_beams=5, early_stopping=True)
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

def correct_grammar(incorrect_sentence, model, tokenizer, device, max_length=128):
    input_text = f"grammar: {incorrect_sentence}"
    inputs = tokenizer([input_text], max_length=max_length, padding='max_length', truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        generated_ids = model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, max_length=max_length, num_beams=5, early_stopping=True)
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

DISFLUENCY_EXPLANATIONS = {
    "um": "This is a filled pause.",
    "uh": "This is a filled pause.",
    "like": "Used as a filler word.",
    "you know": "Used as a filler.",
    "i mean": "A common reformulation marker.",
    "so": "Can be a discourse marker or filler.",
    "and": "Can be a discourse marker or filler.",
}

def get_diff_feedback(original_sentence: str, corrected_sentence: str, correction_stage: str) -> tuple[list, str, str]:
    feedback_messages = []
    original_words = original_sentence.split()
    corrected_words = corrected_sentence.split()
    matcher = difflib.SequenceMatcher(None, original_words, corrected_words)
    diff_output_original = []
    diff_output_corrected = []

    for opcode, i1, i2, j1, j2 in matcher.get_opcodes():
        if opcode == 'equal':
            diff_output_original.extend(original_words[i1:i2])
            diff_output_corrected.extend(corrected_words[j1:j2])
        elif opcode == 'delete':
            deleted_words = original_words[i1:i2]
            diff_output_original.extend([f"{w}" for w in deleted_words])
            for word in deleted_words:
                word_lower = word.lower().strip(',.?!')
                explanation = DISFLUENCY_EXPLANATIONS.get(word_lower, "Word removed to improve clarity.")
                feedback_messages.append(f"- Disfluency removed: '{word}'. {explanation}")
        elif opcode == 'insert':
            added_words = corrected_words[j1:j2]
            diff_output_corrected.extend([f"{w}" for w in added_words])
            for word in added_words:
                feedback_messages.append(f"- Grammatical change: Added '{word}'.")
        elif opcode == 'replace':
            old_words = original_words[i1:i2]
            new_words = corrected_words[j1:j2]
            diff_output_original.extend([f"{w}" for w in old_words])
            diff_output_corrected.extend([f"{w}" for w in new_words])
            old_str = " ".join(old_words)
            new_str = " ".join(new_words)
            feedback_messages.append(f"- Grammatical change: '{old_str}' replaced with '{new_str}'.")

    highlighted_original = " ".join(diff_output_original)
    highlighted_corrected = " ".join(diff_output_corrected)

    return feedback_messages, highlighted_original, highlighted_corrected


def aggressive_post_cleanup(text: str) -> str:
    text = re.sub(r'(\s*,\s*like\s*,\s*|\s*like\s*,\s*|\s*,\s*like\s*|\s+like\s+)', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'^\s*so\s*,', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def chunk_text_by_words(text, chunk_size=6):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


# @title 6. Define the Final Pipeline Function (Unchanged)
def final_correction_pipeline(text_block, dc_model, dc_tokenizer, gec_model, gec_tokenizer, reward_model, reward_tokenizer, device):
    original_text = text_block
    sentences = re.split(r'(?<=[.!?])\s+', original_text)
    dc_corrected_sentences = []
    for sentence in sentences:
        if not sentence.strip():
            continue
        dc_corrected = correct_disfluency(sentence, dc_model, dc_tokenizer, device)
        dc_corrected_sentences.append(dc_corrected)
    combined_dc_text = " ".join(dc_corrected_sentences)
    gec_output = correct_grammar(combined_dc_text, gec_model, gec_tokenizer, device)
    gec_output_sentences = re.split(r'(?<=[.!?])\s+', gec_output)
    final_sentences = []
    for sentence in gec_output_sentences:
        if not sentence.strip():
            continue
        final_corrected = correct_disfluency(sentence, dc_model, dc_tokenizer, device)
        final_sentences.append(final_corrected)
    final_corrected_text = " ".join(final_sentences)
    clean_final_corrected_text = aggressive_post_cleanup(final_corrected_text)
    feedback_messages, highlighted_original, highlighted_corrected = get_diff_feedback(original_text, clean_final_corrected_text, "combined")
    chunks_to_score = chunk_text_by_words(clean_final_corrected_text)
    fluency_scores = []
    for chunk in chunks_to_score:
        inputs_reward = reward_tokenizer([chunk], padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs_reward = reward_model(**inputs_reward)
            probabilities = torch.softmax(outputs_reward.logits, dim=1)[0]
            fluency_scores.append(probabilities[0].item())
    if fluency_scores:
        fluency_score = sum(fluency_scores) / len(fluency_scores)
    else:
        fluency_score = 0.0
    return clean_final_corrected_text, feedback_messages, highlighted_original, highlighted_corrected, fluency_score


# @title 7. Run the complete pipeline with voice input (Updated)
if all([stt_model, dc_model, gec_model, reward_model]):
    # --- Step 1: Record and Transcribe Audio ---
    audio_file = record_audio()
    transcribed_text = transcribe_audio(audio_file, stt_model, stt_processor, device)

    # Proceed only if transcription was successful
    if transcribed_text and not transcribed_text.startswith("["):
        # --- Step 2: Run the Correction Pipeline on the Transcribed Text ---
        print("\n--- Running the correction pipeline on your transcribed text ---")
        final_output, feedback, original_highlighted, final_highlighted, fluency_score = final_correction_pipeline(
            transcribed_text, dc_model, dc_tokenizer, gec_model, gec_tokenizer, reward_model, reward_tokenizer, device
        )

        print("\n" + "="*50)
        print("Transcribed Text (Original):")
        print(transcribed_text)
        print("\nOriginal Text (Highlighted Diff):")
        print(original_highlighted)
        print("\nFinal Corrected Text (Cleaned for Fluency Score):")
        print(final_output)
        print("\nFinal Corrected Text (Highlighted Diff):")
        print(final_highlighted)
        print(f"\nFinal Fluency Score (Averaged): {fluency_score:.2f} (0.00 = Disfluent, 1.00 = Fluent)")
        print("\nFeedback Report:")
        for msg in feedback:
            print(msg)
        print("="*50)
    else:
        print(f"Could not proceed with correction due to an error: {transcribed_text}")

else:
    print("\nOne or more models failed to load. Cannot run the pipeline.")