# Fluent-Talk-A-Speech-to-Speech-Disfluency-Correction-System-for-Enhanced-Communication

## Project Overview

This project presents a comprehensive, multi-stage NLP pipeline designed to automatically correct disfluent and grammatically incorrect spoken language. The system takes raw speech, processes it through a series of specialized models, and produces a fluent, professional-quality text output.

The core innovation is a **cascaded, multi-pass architecture** that breaks down the complex problem of spontaneous speech correction into a series of manageable and highly accurate steps.

## Key Features

* **Multi-Stage Correction:** A robust pipeline that first removes disfluencies and then corrects grammar, leveraging the strengths of two distinct models.

* **Dual-Purpose Models:** The system uses two custom-trained models:

  1. A fine-tuned **T5 model** for precise Disfluency Correction (DC).

  2. A fine-tuned **BERT model** that acts as an automated **Reward Model** to objectively score the fluency and correctness of the final output.

* **Comprehensive Feedback:** A `difflib`-based feedback generation system provides a detailed, word-by-word report of every change made, enhancing the interpretability of the corrections.

* **End-to-End Functionality:** The pipeline is designed to work seamlessly with an Automatic Speech Recognition (ASR) model (e.g., Whisper) to create a complete voice-to-text correction tool.

## Project Architecture

The project's architecture is divided into two phases: an **Offline Training Pipeline** and a **Real-Time Inference Pipeline**.

### Offline Training Pipeline

This phase is a one-time process for creating the specialized models.

* **Data Ingestion:** A custom CSV dataset containing pairs of `Disfluent Sentences` and `Fluent Sentences` is used as the ground truth for training.

* **Model Training:**

  * **Disfluency Correction (DC) Model:** A pre-trained T5 model is fine-tuned on the dataset to specialize in disfluency removal.

  * **Reward Model:** A pre-trained BERT model is fine-tuned as a binary classifier to judge a sentence as either "Fluent" or "Disfluent."

### Real-Time Inference Pipeline

This pipeline represents the full, end-to-end user experience.

1. **ASR (Transcription):** Raw voice input is transcribed into a text string (e.g., using a Whisper model).

2. **DC (Disfluency Correction):** The text is passed through the fine-tuned T5 model to remove fillers and repetitions.

3. **GEC (Grammar Correction):** The cleaned text is passed to a pre-trained GEC model to fix grammatical errors.

4. **Parallel Evaluation:** The final corrected text is sent to two modules in parallel:

   * The **Reward Model** produces an objective fluency score.

   * A **Feedback Generator** compares the original and corrected text to report changes.

5. **Output:** All elements are combined for a comprehensive user-facing result.

## Methodology & Results

The project was implemented in Python using the Hugging Face `transformers` library. The performance of the fine-tuned models was validated on a held-out test set, yielding strong results:

* **Disfluency Correction:** Achieved a **BLEU Score of over 95** and a low **TER Score** on the test set, demonstrating high accuracy in text correction.

* **Reward Model:** The fluency classifier achieved **~93% accuracy**, proving its reliability as an automated quality checker for the pipeline's output.

## Getting Started

To replicate this project, you will need to:

1. Clone this repository.

2. Install the required Python packages (`pip install -r requirements.txt`).

3. Upload your own custom `en-disfluent-sentences-labelled.csv.csv` dataset.

4. Run the provided fine-tuning scripts to train your own models.

5. (Optional) Upload your models to the Hugging Face Hub for easy access and sharing.
