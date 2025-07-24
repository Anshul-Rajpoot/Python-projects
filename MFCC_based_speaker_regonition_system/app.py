import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.fftpack import dct
from datetime import datetime
import uuid
import io
import soundfile as sf
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import the refactored MFCC processing class
from mfcc_module import MFCCProcessor

# Configure page layout
st.set_page_config(layout="wide", page_title="MFCC Feature Extraction System")

# --- Student Information Section ---
st.sidebar.title("Student Information")
st.sidebar.info("Submitted by: Anshul Rajpoot ")
st.sidebar.info("Roll No: 2311401168")

# --- Main Title ---
st.title("MFCC Feature Extraction Pipeline for Speaker Recognition")

# --- File Upload Section ---
uploaded_file = st.file_uploader("Upload a WAV audio file", type=["wav", "wave"])

# --- Parameters Configuration ---
st.sidebar.header("Processing Parameters")

# Frame parameters
frame_size = st.sidebar.slider("Frame size (ms)", 20, 50, 25)
frame_stride = st.sidebar.slider("Overlap Analysis (%)", 0, 100, 10) # Stride as percentage of frame length
# Convert stride percentage to ms for MFCCProcessor
frame_stride_ms = frame_size * (1 - frame_stride / 100.0)
frame_stride_ms = int(round(frame_stride_ms))


# FFT parameters
n_fft = st.sidebar.selectbox("FFT size", [256, 512, 1024], index=1)

# Mel filter parameters
n_mels = st.sidebar.slider("Number of Mel filters", 10, 40, 20)

# MFCC parameters
n_mfcc = st.sidebar.slider("Number of MFCC coefficients", 5, 20, 13)

# Sample rate adjustment
target_sr_display_options = [('Original', 0), (8000, 8000), (16000, 16000), (22050, 22050), (44100, 44100)]
target_sr_selection = st.sidebar.selectbox("Target sample rate (Hz) (0 for Original)", 
                                           options=[opt[0] for opt in target_sr_display_options],
                                           index=2) # Default to 16000 Hz
selected_target_sr_value = [opt[1] for opt in target_sr_display_options if opt[0] == target_sr_selection][0]


# Pre-emphasis coefficient
pre_emphasis = st.sidebar.slider("Pre-emphasis coefficient", 0.90, 0.99, 0.97)

# --- Audio Loading Function ---
@st.cache_data(show_spinner=False)
def load_audio_file(uploaded_file_buffer, target_sr_val):
    """Load audio file with multiple fallback methods and optionally resample."""
    original_signal = None
    original_sr = None
    signal = None
    sr = None

    try:
        audio_bytes = uploaded_file_buffer.read()
        uploaded_file_buffer.seek(0) # Reset pointer for potential re-reads

        # Try librosa.load from bytes first
        try:
            temp_signal, temp_sr = librosa.load(io.BytesIO(audio_bytes), sr=None) # Load at original SR first
            original_signal = temp_signal
            original_sr = temp_sr
        except Exception as e_librosa:
            st.warning(f"Librosa initial load failed: {e_librosa}. Trying soundfile.")
            # Fallback to soundfile if librosa fails directly from BytesIO
            with io.BytesIO(audio_bytes) as f:
                temp_signal, temp_sr = sf.read(f)
            if len(temp_signal.shape) > 1: # If stereo, take first channel
                temp_signal = temp_signal[:, 0]
            original_signal = temp_signal
            original_sr = temp_sr
        
        signal = original_signal
        sr = original_sr

        if target_sr_val != 0 and sr != target_sr_val:
            signal = librosa.resample(signal, orig_sr=sr, target_sr=target_sr_val)
            sr = target_sr_val
        
        return original_signal, original_sr, signal, sr
        
    except Exception as e:
        st.error(f"Failed to load audio file: {str(e)}")
        return None, None, None, None

# --- Main Processing ---
if uploaded_file is not None:
    # Use st.spinner for a smoother UX during file loading and processing
    with st.spinner("Loading and processing audio..."):
        # We need a new buffer for each call to load_audio_file if we want to ensure it's read from the beginning
        # or if st.cache_data handles the file uploader object's internal state
        # For simplicity, we'll re-read the bytes if needed by the caching mechanism
        
        # Create a new BytesIO object for loading, as the original uploaded_file might be consumed
        uploaded_file_content_for_load = io.BytesIO(uploaded_file.getvalue())
        
        original_signal, original_sr, processed_signal, processed_sr = load_audio_file(uploaded_file_content_for_load, selected_target_sr_value)
        
        if processed_signal is None:
            st.stop()
        
    st.write(f"Original Audio duration: {len(original_signal)/original_sr:.2f} seconds | Original Sample rate: {original_sr} Hz")
    if selected_target_sr_value != 0:
        st.write(f"Processed Audio (resampled) duration: {len(processed_signal)/processed_sr:.2f} seconds | Processed Sample rate: {processed_sr} Hz")
    else:
        st.write(f"Processed Audio (original SR) duration: {len(processed_signal)/processed_sr:.2f} seconds | Processed Sample rate: {processed_sr} Hz")

    # Initialize MFCCProcessor with current sidebar parameters
    mfcc_processor = MFCCProcessor(
        sr=processed_sr,
        frame_size=frame_size,
        frame_stride=frame_stride_ms, # Use calculated stride in ms
        n_fft=n_fft,
        n_mels=n_mels,
        n_mfcc=n_mfcc,
        pre_emphasis=pre_emphasis
    )
    
    with st.spinner("Computing MFCC features..."):
        try:
            result = mfcc_processor.full_pipeline(processed_signal, processed_sr)
        except Exception as e:
            st.error(f"An error occurred during MFCC computation: {str(e)}")
            st.stop()
    
    # --- Visualizations ---
    st.subheader("Time Domain Signal")
    st.pyplot(mfcc_processor.plot_time_domain(processed_signal, processed_sr, title="Processed Time Domain Signal"))
    
    # Spectrogram vs MFCC comparison 
    st.subheader("Spectrogram vs MFCC Comparison")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Spectrogram**")
        st.pyplot(mfcc_processor.plot_spectrogram(processed_signal, processed_sr, title="Spectrogram"))
    
    with col2:
        st.markdown("**MFCC Heatmap**")
        st.pyplot(mfcc_processor.plot_mfcc(result['mfcc'], processed_sr, title="MFCC Coefficients Heatmap"))
    
    # Mel filter bank visualization 
    st.subheader("Mel Filter Bank")
    st.pyplot(mfcc_processor.plot_mel_filterbank(result['mel_fbanks'], title="Mel Filter Bank (based on selected parameters)"))
    
    # --- Parameter Analysis ---
    st.subheader("Parameter Analysis and Impact Study")
    
    # 2. Interactive Frame Size and Overlap Analysis [cite: 176, 177]
    st.markdown("### Effect of Frame Size and Overlap")
    st.info("The MFCC heatmap and Spectrogram above dynamically update with 'Frame size' and 'Overlap Analysis' sliders.")
    st.markdown("Below are static comparisons showing how different *fixed* values of frame size and overlap can affect the resulting MFCCs. Observe changes in time-frequency resolution.")

    st.markdown("**Impact of Frame Size (fixed overlap)**")
    frame_sizes_for_analysis = [20, 30, 50] # in ms
    cols_frame_size_analysis = st.columns(len(frame_sizes_for_analysis))
    
    for i, fs in enumerate(frame_sizes_for_analysis):
        with cols_frame_size_analysis[i]:
            st.markdown(f"**{fs}ms Frame Size**")
            mfcc_processor_temp = MFCCProcessor(
                sr=processed_sr, frame_size=fs, frame_stride=frame_stride_ms, # Keep current stride for comparison
                n_fft=n_fft, n_mels=n_mels, n_mfcc=n_mfcc, pre_emphasis=pre_emphasis
            )
            res_temp = mfcc_processor_temp.full_pipeline(processed_signal, processed_sr)
            fig_mfcc_temp = mfcc_processor_temp.plot_mfcc(res_temp['mfcc'], processed_sr, title=f"MFCC - {fs}ms Frame")
            st.pyplot(fig_mfcc_temp)

    st.markdown("**Impact of Overlap (fixed frame size)**")
    # For overlap, let's use percentage of frame length for clarity
    overlap_percentages_for_analysis = [25, 50, 75] 
    
    cols_overlap_analysis = st.columns(len(overlap_percentages_for_analysis))

    for i, ovl_pct in enumerate(overlap_percentages_for_analysis):
        with cols_overlap_analysis[i]:
            st.markdown(f"**{ovl_pct}% Overlap**")
            # Calculate frame_stride_ms for this specific overlap percentage
            temp_frame_stride_ms = frame_size * (1 - ovl_pct / 100.0)
            temp_frame_stride_ms = int(round(temp_frame_stride_ms))
            
            mfcc_processor_temp = MFCCProcessor(
                sr=processed_sr, frame_size=frame_size, frame_stride=temp_frame_stride_ms, 
                n_fft=n_fft, n_mels=n_mels, n_mfcc=n_mfcc, pre_emphasis=pre_emphasis
            )
            res_temp = mfcc_processor_temp.full_pipeline(processed_signal, processed_sr)
            fig_mfcc_temp = mfcc_processor_temp.plot_mfcc(res_temp['mfcc'], processed_sr, title=f"MFCC - {ovl_pct}% Overlap")
            st.pyplot(fig_mfcc_temp)


    # 3. Mel Filter Bank Customization and Visualization [cite: 179, 180]
    st.markdown("### Effect of Number of Mel Filters")
    st.info("The Mel Filter Bank plot and MFCC heatmap above dynamically update with the 'Number of Mel filters' slider.")
    st.markdown("Below are static comparisons showing the Mel filter bank shape and resulting MFCCs for different *fixed* numbers of filters.")

    mel_counts_for_analysis = [10, 20, 30]
    cols_mel_count_analysis = st.columns(len(mel_counts_for_analysis))
    
    for i, mc in enumerate(mel_counts_for_analysis):
        with cols_mel_count_analysis[i]:
            st.markdown(f"**{mc} Mel Filters**")
            mfcc_processor_temp = MFCCProcessor(
                sr=processed_sr, frame_size=frame_size, frame_stride=frame_stride_ms,
                n_fft=n_fft, n_mels=mc, n_mfcc=n_mfcc, pre_emphasis=pre_emphasis
            )
            res_temp = mfcc_processor_temp.full_pipeline(processed_signal, processed_sr)
            
            # Plot Mel Filter Bank
            fig_mel_fb = mfcc_processor_temp.plot_mel_filterbank(res_temp['mel_fbanks'], title=f"{mc} Mel FBs")
            st.pyplot(fig_mel_fb)
            
            # Plot MFCCs
            fig_mfcc_mc = mfcc_processor_temp.plot_mfcc(res_temp['mfcc'], processed_sr, title=f"MFCC - {mc} Mels")
            st.pyplot(fig_mfcc_mc)


    # 4. Dynamic Sampling Rate Adjustment and MFCC Degradation Study 
    if selected_target_sr_value != 0:
        st.markdown("### Sampling Rate Adjustment and Degradation Study")
        st.info("Observe the perceptual and feature-level impact of resampling by comparing the original and processed audio below.")
        
        st.markdown("#### Time Domain Signal Comparison")
        col_orig_time, col_proc_time = st.columns(2)
        with col_orig_time:
            st.markdown("**Original Signal**")
            st.pyplot(mfcc_processor.plot_time_domain(original_signal, original_sr, title="Original Time Domain"))
        with col_proc_time:
            st.markdown("**Resampled Signal**")
            st.pyplot(mfcc_processor.plot_time_domain(processed_signal, processed_sr, title=f"Resampled to {processed_sr}Hz"))

        st.markdown("#### Spectrogram Comparison")
        col_orig_spec, col_proc_spec = st.columns(2)
        with col_orig_spec:
            st.markdown("**Original Spectrogram**")
            # Create a temporary MFCCProcessor for original signal if needed, or directly plot
            st.pyplot(mfcc_processor.plot_spectrogram(original_signal, original_sr, title="Original Spectrogram"))
        with col_proc_spec:
            st.markdown(f"**Resampled Spectrogram ({processed_sr}Hz)**")
            st.pyplot(mfcc_processor.plot_spectrogram(processed_signal, processed_sr, title=f"Resampled Spectrogram ({processed_sr}Hz)"))

        st.markdown("#### MFCC Heatmap Comparison")
        col_orig_mfcc, col_proc_mfcc = st.columns(2)
        with col_orig_mfcc:
            st.markdown("**Original MFCCs**")
            # Need to compute MFCCs for the original signal as well for comparison
            mfcc_processor_orig = MFCCProcessor(
                sr=original_sr, frame_size=frame_size, frame_stride=frame_stride_ms,
                n_fft=n_fft, n_mels=n_mels, n_mfcc=n_mfcc, pre_emphasis=pre_emphasis
            )
            result_orig = mfcc_processor_orig.full_pipeline(original_signal, original_sr)
            st.pyplot(mfcc_processor.plot_mfcc(result_orig['mfcc'], original_sr, title="Original MFCCs"))
        with col_proc_mfcc:
            st.markdown(f"**Resampled MFCCs ({processed_sr}Hz)**")
            st.pyplot(mfcc_processor.plot_mfcc(result['mfcc'], processed_sr, title=f"Resampled MFCCs ({processed_sr}Hz)"))
    elif selected_target_sr_value == 0:
        st.markdown("### Sampling Rate Adjustment and Degradation Study")
        st.info("The audio is currently processed at its original sample rate. Select a target sample rate from the sidebar to observe resampling effects.")
        st.markdown("To analyze degradation, select a specific target sample rate (e.g., 8000 Hz, 16000 Hz) from the 'Target sample rate (Hz)' dropdown in the sidebar. This will enable comparison plots.")


# --- Execution Information ---
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Execution Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.sidebar.markdown(f"**Session ID:** {str(uuid.uuid4())[:8]}")