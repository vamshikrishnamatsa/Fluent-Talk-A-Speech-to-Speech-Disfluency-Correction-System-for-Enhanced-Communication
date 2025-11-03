# -*- coding: utf-8 -*-
"""
Speech Fluency Correction - Enhanced Gradio App with Better Audio Support
Run with: python app.py
"""

import gradio as gr
import torch
from transformers import (
    T5ForConditionalGeneration, T5Tokenizer, AutoModelForSeq2SeqLM, AutoTokenizer,
    AutoModelForSequenceClassification, WhisperForConditionalGeneration, WhisperProcessor
)
import re
import difflib
import librosa
import numpy as np
import warnings
import soundfile as sf
import tempfile
import os
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
HF_USERNAME = "vamshi0310"
DC_REPO_ID = f"{HF_USERNAME}/fine-tuned-disfluency-model"
REWARD_REPO_ID = f"{HF_USERNAME}/custom-fluency-classifier"
GEC_MODEL_ID = 'vennify/t5-base-grammar-correction'
STT_MODEL_ID = "openai/whisper-base"
HF_TOKEN = "hf token"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# ==================== LOAD MODELS ====================
print("📦 Loading models... This may take 10-20 minutes on first run.")

try:
    print("  ➤ Loading Speech-to-Text model...")
    stt_processor = WhisperProcessor.from_pretrained(STT_MODEL_ID, token=HF_TOKEN)
    stt_model = WhisperForConditionalGeneration.from_pretrained(STT_MODEL_ID, token=HF_TOKEN).to(device)
    stt_model.eval()

    print("  ➤ Loading Disfluency Correction model...")
    dc_tokenizer = AutoTokenizer.from_pretrained(DC_REPO_ID, token=HF_TOKEN)
    dc_model = T5ForConditionalGeneration.from_pretrained(DC_REPO_ID, token=HF_TOKEN).to(device)
    dc_model.eval()

    print("  ➤ Loading Grammar Correction model...")
    gec_tokenizer = AutoTokenizer.from_pretrained(GEC_MODEL_ID, token=HF_TOKEN)
    gec_model = AutoModelForSeq2SeqLM.from_pretrained(GEC_MODEL_ID, token=HF_TOKEN).to(device)
    gec_model.eval()

    print("  ➤ Loading Reward Model...")
    reward_tokenizer = AutoTokenizer.from_pretrained(REWARD_REPO_ID, token=HF_TOKEN)
    reward_model = AutoModelForSequenceClassification.from_pretrained(REWARD_REPO_ID, token=HF_TOKEN).to(device)
    reward_model.eval()

    print("✅ All models loaded successfully!\n")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    raise

# ==================== HELPER FUNCTIONS ====================

DISFLUENCY_EXPLANATIONS = {
    "um": "Filled pause removed for clarity",
    "uh": "Filled pause removed for clarity",
    "like": "Filler word removed",
    "you know": "Filler phrase removed",
    "i mean": "Reformulation marker removed",
    "so": "Discourse marker removed",
    "and": "Discourse marker removed",
    "well": "Discourse marker removed",
    "actually": "Hedge word removed",
    "basically": "Hedge word removed",
}

def preprocess_audio(audio_input):
    """Preprocess audio input to ensure compatibility."""
    if audio_input is None:
        return None
    
    try:
        # Handle tuple input (sample_rate, audio_data)
        if isinstance(audio_input, tuple):
            sample_rate, audio_data = audio_input
            
            # Convert to float32 if needed
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            
            # Convert stereo to mono if needed
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
            
            return audio_data
        
        # Handle file path input
        elif isinstance(audio_input, str):
            audio_data, sample_rate = librosa.load(audio_input, sr=16000)
            return audio_data
        
        return None
    
    except Exception as e:
        print(f"Audio preprocessing error: {e}")
        return None

def transcribe_audio(audio_input):
    """Transcribe audio to text using Whisper with improved handling."""
    if audio_input is None:
        return "❌ No audio provided"
    
    try:
        # Preprocess audio
        audio_array = preprocess_audio(audio_input)
        
        if audio_array is None:
            return "❌ Failed to process audio"
        
        # Check audio length
        if len(audio_array) < 1600:  # Less than 0.1 seconds
            return "❌ Audio too short. Please record at least 1 second."
        
        # Process with Whisper
        input_features = stt_processor(
            audio_array, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_features.to(device)
        
        with torch.no_grad():
            predicted_ids = stt_model.generate(
                input_features,
                max_length=448,
                num_beams=5,
                temperature=0.0
            )
        
        transcription = stt_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        # Check if transcription is empty or too short
        if not transcription or len(transcription.strip()) < 3:
            return "❌ Could not transcribe audio. Please speak clearly and try again."
        
        return transcription.strip()
    
    except Exception as e:
        print(f"Transcription error: {e}")
        return f"❌ Transcription error: {str(e)}"

def correct_disfluency(sentence, max_length=128):
    """Correct disfluencies in a sentence."""
    if not sentence or len(sentence.strip()) < 2:
        return sentence
    
    try:
        input_text = f"correct disfluency: {sentence}"
        inputs = dc_tokenizer([input_text], max_length=max_length, padding='max_length', 
                             truncation=True, return_tensors="pt").to(device)
        
        with torch.no_grad():
            generated_ids = dc_model.generate(
                input_ids=inputs.input_ids, 
                attention_mask=inputs.attention_mask, 
                max_length=max_length, 
                num_beams=5, 
                early_stopping=True
            )
        
        return dc_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Disfluency correction error: {e}")
        return sentence

def correct_grammar(sentence, max_length=128):
    """Correct grammar in a sentence."""
    if not sentence or len(sentence.strip()) < 2:
        return sentence
    
    try:
        input_text = f"grammar: {sentence}"
        inputs = gec_tokenizer([input_text], max_length=max_length, padding='max_length', 
                              truncation=True, return_tensors="pt").to(device)
        
        with torch.no_grad():
            generated_ids = gec_model.generate(
                input_ids=inputs.input_ids, 
                attention_mask=inputs.attention_mask, 
                max_length=max_length, 
                num_beams=5, 
                early_stopping=True
            )
        
        return gec_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Grammar correction error: {e}")
        return sentence

def get_diff_feedback(original_sentence, corrected_sentence):
    """Generate feedback on corrections made."""
    feedback_messages = []
    original_words = original_sentence.split()
    corrected_words = corrected_sentence.split()
    matcher = difflib.SequenceMatcher(None, original_words, corrected_words)
    
    for opcode, i1, i2, j1, j2 in matcher.get_opcodes():
        if opcode == 'delete':
            deleted_words = original_words[i1:i2]
            for word in deleted_words:
                word_lower = word.lower().strip(',.?!')
                explanation = DISFLUENCY_EXPLANATIONS.get(word_lower, "Word removed to improve clarity")
                feedback_messages.append(f"❌ Removed '{word}': {explanation}")
        elif opcode == 'replace':
            old_words = " ".join(original_words[i1:i2])
            new_words = " ".join(corrected_words[j1:j2])
            feedback_messages.append(f"✏️ Changed '{old_words}' → '{new_words}'")
        elif opcode == 'insert':
            added_words = " ".join(corrected_words[j1:j2])
            feedback_messages.append(f"➕ Added '{added_words}' for grammatical correctness")
    
    return "\n".join(feedback_messages) if feedback_messages else "✅ No corrections needed - your text is already fluent!"

def aggressive_post_cleanup(text):
    """Clean up remaining disfluencies."""
    text = re.sub(r'(\s*,\s*like\s*,\s*|\s*like\s*,\s*|\s*,\s*like\s*|\s+like\s+)', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'^\s*so\s*,', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def chunk_text_by_words(text, chunk_size=6):
    """Chunk text into smaller pieces for scoring."""
    words = text.split()
    if not words:
        return []
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def calculate_fluency_score(text):
    """Calculate fluency score for text."""
    if not text or len(text.strip()) < 3:
        return 0.0
    
    try:
        chunks = chunk_text_by_words(text)
        if not chunks:
            return 0.0
        
        fluency_scores = []
        
        for chunk in chunks:
            inputs = reward_tokenizer([chunk], padding=True, truncation=True, 
                                     max_length=128, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = reward_model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)[0]
                fluency_scores.append(probabilities[0].item())
        
        if fluency_scores:
            return sum(fluency_scores) / len(fluency_scores)
        return 0.0
    except Exception as e:
        print(f"Fluency score error: {e}")
        return 0.0

# ==================== MAIN CORRECTION PIPELINE ====================

def correction_pipeline(text):
    """Main correction pipeline for text input."""
    if not text or not text.strip():
        return "❌ Please provide some text", "", 0, "⚠️ No input"
    
    original_text = text.strip()
    
    try:
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', original_text)
        
        # First pass: Disfluency correction
        dc_corrected_sentences = []
        for sentence in sentences:
            if sentence.strip():
                dc_corrected = correct_disfluency(sentence)
                dc_corrected_sentences.append(dc_corrected)
        
        combined_dc_text = " ".join(dc_corrected_sentences)
        
        # Grammar correction
        gec_output = correct_grammar(combined_dc_text)
        
        # Second pass: Disfluency correction
        gec_output_sentences = re.split(r'(?<=[.!?])\s+', gec_output)
        final_sentences = []
        for sentence in gec_output_sentences:
            if sentence.strip():
                final_corrected = correct_disfluency(sentence)
                final_sentences.append(final_corrected)
        
        final_corrected_text = " ".join(final_sentences)
        clean_final_corrected_text = aggressive_post_cleanup(final_corrected_text)
        
        # Generate feedback
        feedback = get_diff_feedback(original_text, clean_final_corrected_text)
        
        # Calculate fluency score
        fluency_score = calculate_fluency_score(clean_final_corrected_text)
        fluency_percentage = fluency_score * 100
        
        # Create score interpretation
        if fluency_score >= 0.8:
            score_text = f"🌟 Excellent! ({fluency_percentage:.1f}%)"
        elif fluency_score >= 0.6:
            score_text = f"👍 Good ({fluency_percentage:.1f}%)"
        else:
            score_text = f"⚠️ Needs improvement ({fluency_percentage:.1f}%)"
        
        return clean_final_corrected_text, feedback, fluency_percentage, score_text
    
    except Exception as e:
        print(f"Pipeline error: {e}")
        return f"❌ Error: {str(e)}", "", 0, "⚠️ Error"

def process_audio(audio_input):
    """Process audio file: transcribe and correct."""
    if audio_input is None:
        return "❌ No audio provided", "", "", 0, "⚠️ No audio"
    
    # Transcribe
    print("Transcribing audio...")
    transcription = transcribe_audio(audio_input)
    
    if transcription.startswith("❌"):
        return transcription, "", "", 0, "⚠️ Error"
    
    # Correct
    print("Correcting transcription...")
    corrected_text, feedback, fluency_percentage, score_text = correction_pipeline(transcription)
    
    return transcription, corrected_text, feedback, fluency_percentage, score_text

# ==================== GRADIO INTERFACE ====================

def create_interface():
    """Create Gradio interface with improved audio support."""
    
    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
    }
    .audio-container {
        border: 2px dashed #4F46E5;
        border-radius: 8px;
        padding: 20px;
        background: #F9FAFB;
    }
    """
    
    with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="Speech Fluency Correction") as demo:
        gr.Markdown(
            """
            # 🎙️ Speech Fluency Correction System
            ### Remove disfluencies and improve your speech clarity with AI
            
            **Powered by Whisper AI, T5, and Custom Models**
            """
        )
        
        with gr.Tabs():
            # Tab 1: Audio Input - ENHANCED
            with gr.Tab("🎤 Audio Input"):
                gr.Markdown(
                    """
                    ### 🎤 Live Audio Recording
                    **Instructions:**
                    1. Click the **microphone icon** to start recording
                    2. Speak clearly (at least 1-2 seconds)
                    3. Click **stop** when done
                    4. Click **🚀 Process Audio** button
                    
                    Or **upload** an audio file (MP3, WAV, M4A, etc.)
                    """
                )
                
                with gr.Row():
                    audio_input = gr.Audio(
                        sources=["microphone", "upload"],
                        type="numpy",  # Changed to numpy for better compatibility
                        label="🎤 Record Your Speech or Upload Audio File",
                        elem_classes="audio-container"
                    )
                
                with gr.Row():
                    audio_button = gr.Button("🚀 Process Audio", variant="primary", size="lg", scale=2)
                    clear_audio = gr.Button("🔄 Clear", variant="secondary", size="lg", scale=1)
                
                gr.Markdown("---")
                
                with gr.Row():
                    with gr.Column():
                        transcription_output = gr.Textbox(
                            label="📝 Original Transcription",
                            lines=5,
                            placeholder="Your transcribed speech will appear here...",
                            show_copy_button=True
                        )
                    
                    with gr.Column():
                        audio_corrected_output = gr.Textbox(
                            label="✨ Corrected Text",
                            lines=5,
                            placeholder="Corrected version will appear here...",
                            show_copy_button=True
                        )
                
                with gr.Row():
                    with gr.Column(scale=3):
                        audio_fluency_slider = gr.Slider(
                            minimum=0,
                            maximum=100,
                            value=0,
                            label="🎯 Fluency Score (%)",
                            interactive=False,
                            info="Higher is better"
                        )
                    
                    with gr.Column(scale=1):
                        audio_score_text = gr.Textbox(
                            label="Rating",
                            lines=1,
                            interactive=False
                        )
                
                audio_feedback_output = gr.Textbox(
                    label="💡 Detailed Feedback",
                    lines=8,
                    placeholder="Correction details will appear here...",
                    show_copy_button=True
                )
                
                # Button actions
                audio_button.click(
                    fn=process_audio,
                    inputs=[audio_input],
                    outputs=[transcription_output, audio_corrected_output, audio_feedback_output, 
                            audio_fluency_slider, audio_score_text]
                )
                
                clear_audio.click(
                    fn=lambda: (None, "", "", "", 0, ""),
                    inputs=[],
                    outputs=[audio_input, transcription_output, audio_corrected_output, 
                            audio_feedback_output, audio_fluency_slider, audio_score_text]
                )
            
            # Tab 2: Text Input
            with gr.Tab("⌨️ Text Input"):
                gr.Markdown(
                    """
                    ### ⌨️ Direct Text Correction
                    Type or paste your text below for instant correction.
                    """
                )
                
                text_input = gr.Textbox(
                    label="📝 Enter Your Text",
                    placeholder="Example: Um, so like, I think we should, uh, you know, meet tomorrow at like 3 PM.",
                    lines=5
                )
                
                with gr.Row():
                    text_button = gr.Button("🚀 Correct Text", variant="primary", size="lg", scale=2)
                    clear_text = gr.Button("🔄 Clear", variant="secondary", size="lg", scale=1)
                
                gr.Markdown("---")
                
                text_corrected_output = gr.Textbox(
                    label="✨ Corrected Text",
                    lines=5,
                    placeholder="Corrected version will appear here...",
                    show_copy_button=True
                )
                
                with gr.Row():
                    with gr.Column(scale=3):
                        text_fluency_slider = gr.Slider(
                            minimum=0,
                            maximum=100,
                            value=0,
                            label="🎯 Fluency Score (%)",
                            interactive=False,
                            info="Higher is better"
                        )
                    
                    with gr.Column(scale=1):
                        text_score_text = gr.Textbox(
                            label="Rating",
                            lines=1,
                            interactive=False
                        )
                
                text_feedback_output = gr.Textbox(
                    label="💡 Detailed Feedback",
                    lines=8,
                    placeholder="Correction details will appear here...",
                    show_copy_button=True
                )
                
                # Button actions
                text_button.click(
                    fn=correction_pipeline,
                    inputs=[text_input],
                    outputs=[text_corrected_output, text_feedback_output, 
                            text_fluency_slider, text_score_text]
                )
                
                clear_text.click(
                    fn=lambda: ("", "", "", 0, ""),
                    inputs=[],
                    outputs=[text_input, text_corrected_output, text_feedback_output,
                            text_fluency_slider, text_score_text]
                )
                
                # Examples
                gr.Markdown("### 📚 Try These Examples:")
                gr.Examples(
                    examples=[
                        ["Um, so like, I think we should, uh, you know, meet tomorrow at like 3 PM."],
                        ["I mean, uh, the project is, like, almost done, you know, and we just need to, um, finish testing."],
                        ["So, um, we need to, like, finish this by, uh, next week, basically."],
                        ["Well, actually, I was thinking, um, that we could, you know, try a different approach."],
                        ["Like, the thing is, uh, we really need to, I mean, get this done soon."]
                    ],
                    inputs=text_input,
                    label="Click any example to try it"
                )
        
        gr.Markdown(
            """
            ---
            ### ℹ️ How It Works:
            
            1. **Speech-to-Text**: Converts audio to text using OpenAI's Whisper model
            2. **Disfluency Detection**: Identifies and removes filler words (um, uh, like, you know, etc.)
            3. **Grammar Correction**: Fixes grammatical errors and improves sentence structure
            4. **Fluency Scoring**: Evaluates the final text quality (0-100%)
            
            ### 💻 System Information:
            - **Device**: {}
            - **Models**: All loaded and ready
            - **Audio Format**: Supports MP3, WAV, M4A, WebM, and more
            - **Sample Rate**: Automatically resampled to 16kHz
            
            ### 🎯 Tips for Best Results:
            - Speak clearly and at a normal pace
            - Record for at least 2-3 seconds
            - Avoid background noise if possible
            - Use a good quality microphone
            
            ### 🔧 Troubleshooting:
            - If recording doesn't work, check browser permissions
            - If transcription is poor, try re-recording with clearer speech
            - First run may take time to download models (~5GB)
            """.format(device)
        )
    
    return demo

# ==================== LAUNCH APP ====================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Starting Enhanced Gradio App...")
    print("="*60)
    print("\n💡 Tips:")
    print("  • First run will download models (~5GB)")
    print("  • Use Chrome/Edge for best audio recording")
    print("  • Allow microphone permissions when prompted")
    print("  • Speak clearly for 2-3 seconds minimum")
    print("\n" + "="*60 + "\n")
    
    demo = create_interface()
    demo.launch(
        share=False,  # Set to True to create public shareable link
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        show_api=False
    )