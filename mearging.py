import os
import sys
import warnings

# Set environment variables BEFORE any other imports
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_SILENCE_TF_CPP_MIN_LOG_LEVEL'] = '3'

# Suppress all warnings
warnings.filterwarnings('ignore')

# Redirect stderr to suppress low-level warnings
from contextlib import redirect_stderr
import io

# Import TensorFlow and set logger level
try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except ImportError:
    tf = None

import streamlit as st
import streamlit.components.v1 as components
import parselmouth
import numpy as np
import pandas as pd
from scipy.stats import entropy
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tempfile
import cv2
import mediapipe as mp
import time
import base64
from io import StringIO

# Optional imports with fallbacks
try:
    from pydub import AudioSegment
    from pydub.utils import which
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    st.warning("⚠️ pydub not available. Audio conversion will be limited to WAV files.")

try:
    import pickle
    import joblib
    from sklearn.preprocessing import MinMaxScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.warning("⚠️ scikit-learn not available. ML predictions will be disabled.")

# --- Shared Utilities and State Management ---
@st.cache_resource(show_spinner=False)
def load_rf_model():
    """Load the trained Random Forest model and scaler"""
    if not SKLEARN_AVAILABLE:
        return None, None
        
    try:
        model_path = 'random_forest_model.pkl'
        if not os.path.exists(model_path):
            st.info('📝 Note: Trained model file random_forest_model.pkl not found. ML predictions will be unavailable.')
            return None, None
        
        model = joblib.load(model_path)
        
        # We need a scaler, which is often saved with the model.
        # If not, we can create a placeholder.
        scaler = MinMaxScaler()
        
        return model, scaler
    except Exception as e:
        st.error(f'Error loading model: {str(e)}')
        return None, None

# Load model and scaler once
rf_model, rf_scaler = load_rf_model() if SKLEARN_AVAILABLE else (None, None)

# --- Voice Analysis Functions ---

def calculate_rpde(signal, m=5, r=0.2):
    """Calculate Recurrence Period Density Entropy (RPDE)"""
    try:
        N = len(signal)
        if N < 100:
            return np.nan
        tau = 1
        embedded = np.array([signal[i:i+m] for i in range(N-m+1)])
        distances = np.sqrt(np.sum((embedded[:, None] - embedded) ** 2, axis=2))
        recurrence_matrix = distances < r * np.std(signal)
        periods = []
        for i in range(len(recurrence_matrix)):
            recurrent_points = np.where(recurrence_matrix[i])[0]
            if len(recurrent_points) > 1:
                periods.extend(np.diff(recurrent_points))
        if len(periods) == 0:
            return np.nan
        hist, _ = np.histogram(periods, bins=50)
        prob = hist / np.sum(hist)
        prob = prob[prob > 0]
        return entropy(prob)
    except:
        return np.nan

def calculate_dfa(signal, min_box_size=4, max_box_size=None):
    """Calculate Detrended Fluctuation Analysis (DFA)"""
    try:
        if max_box_size is None:
            max_box_size = len(signal) // 4
        signal = np.array(signal)
        N = len(signal)
        if N < 16:
            return np.nan
        y = np.cumsum(signal - np.mean(signal))
        box_sizes = np.logspace(np.log10(min_box_size), np.log10(max_box_size), 10).astype(int)
        box_sizes = np.unique(box_sizes)
        fluctuations = []
        for box_size in box_sizes:
            if box_size >= N:
                continue
            n_boxes = N // box_size
            boxes = y[:n_boxes * box_size].reshape(n_boxes, box_size)
            trends = []
            for box in boxes:
                x = np.arange(len(box))
                trend = np.polyfit(x, box, 1)
                trends.append(np.polyval(trend, x))
            trends = np.array(trends).flatten()
            detrended = y[:len(trends)] - trends
            fluctuation = np.sqrt(np.mean(detrended ** 2))
            fluctuations.append(fluctuation)
        if len(fluctuations) < 3:
            return np.nan
        log_box_sizes = np.log10(box_sizes[:len(fluctuations)])
        log_fluctuations = np.log10(fluctuations)
        slope, _ = np.polyfit(log_box_sizes, log_fluctuations, 1)
        return slope
    except:
        return np.nan

def calculate_ppe(f0_values):
    """Calculate Pitch Period Entropy (PPE)"""
    try:
        f0_values = f0_values[~np.isnan(f0_values)]
        if len(f0_values) < 10:
            return np.nan
        periods = 1.0 / f0_values
        period_diffs = np.diff(periods)
        relative_diffs = np.abs(period_diffs) / periods[:-1]
        hist, _ = np.histogram(relative_diffs, bins=50)
        prob = hist / np.sum(hist)
        prob = prob[prob > 0]
        return entropy(prob)
    except:
        return np.nan

def calculate_jitter_mathematical(f0_values):
    """Calculate jitter features using mathematical formulas"""
    try:
        f0_values = f0_values[~np.isnan(f0_values)]
        if len(f0_values) < 10:
            return {
                "Jitter(%)": np.nan, "Jitter(Abs)": np.nan,
                "Jitter:RAP": np.nan, "Jitter:PPQ5": np.nan,
                "Jitter:DDP": np.nan
            }
        periods = 1.0 / f0_values
        N = len(periods)
        period_diffs = np.abs(np.diff(periods))
        mean_period = np.mean(periods)
        jitter_local = np.mean(period_diffs) / mean_period
        jitter_percent = jitter_local * 100
        jitter_absolute = np.mean(period_diffs)
        if N >= 3:
            rap_sum = np.sum(np.abs(periods[1:-1] - (periods[0:-2] + periods[1:-1] + periods[2:]) / 3))
            jitter_rap = (rap_sum / (N - 2)) / mean_period
        else:
            jitter_rap = np.nan
        if N >= 5:
            ppq5_sum = np.sum(np.abs(periods[2:-2] - (periods[0:-4] + periods[1:-3] + periods[2:-2] + periods[3:-1] + periods[4:]) / 5))
            jitter_ppq5 = (ppq5_sum / (N - 4)) / mean_period
        else:
            jitter_ppq5 = np.nan
        if N >= 3:
            first_diffs = np.diff(periods)
            second_diffs = np.diff(first_diffs)
            jitter_ddp = np.mean(np.abs(second_diffs)) / mean_period
        else:
            jitter_ddp = np.nan
        return {
            "Jitter(%)": jitter_percent, "Jitter(Abs)": jitter_absolute,
            "Jitter:RAP": jitter_rap, "Jitter:PPQ5": jitter_ppq5,
            "Jitter:DDP": jitter_ddp
        }
    except Exception as e:
        return {
            "Jitter(%)": np.nan, "Jitter(Abs)": np.nan,
            "Jitter:RAP": np.nan, "Jitter:PPQ5": np.nan,
            "Jitter:DDP": np.nan
        }

def calculate_spread1(f0_values):
    """Calculate spread1 - nonlinear fundamental frequency variation measure"""
    try:
        f0_values = f0_values[~np.isnan(f0_values)]
        if len(f0_values) < 10:
            return np.nan
        mean_f0 = np.mean(f0_values)
        std_f0 = np.std(f0_values)
        return std_f0 / mean_f0 if mean_f0 > 0 else np.nan
    except:
        return np.nan

def calculate_spread2(f0_values):
    """Calculate spread2 - another nonlinear fundamental frequency variation measure"""
    try:
        f0_values = f0_values[~np.isnan(f0_values)]
        if len(f0_values) < 10:
            return np.nan
        q75, q25 = np.percentile(f0_values, [75, 25])
        median_f0 = np.median(f0_values)
        return (q75 - q25) / median_f0 if median_f0 > 0 else np.nan
    except:
        return np.nan

def calculate_D2(f0_values):
    """Calculate D2 - correlation dimension, a measure of fractal dimension"""
    try:
        f0_values = f0_values[~np.isnan(f0_values)]
        if len(f0_values) < 50:
            return np.nan
        embedding_dim = 3
        delay = 1
        if len(f0_values) < embedding_dim * delay + 1:
            return np.nan
        embedded = []
        for i in range(len(f0_values) - (embedding_dim - 1) * delay):
            vector = [f0_values[i + j * delay] for j in range(embedding_dim)]
            embedded.append(vector)
        embedded = np.array(embedded)
        n_points = len(embedded)
        if n_points < 10:
            return np.nan
        radii = np.logspace(-3, 0, 10)
        correlations = []
        for r in radii:
            count = 0
            for i in range(n_points):
                for j in range(i + 1, n_points):
                    distance = np.linalg.norm(embedded[i] - embedded[j])
                    if distance < r:
                        count += 1
            correlation = 2.0 * count / (n_points * (n_points - 1)) if n_points > 1 else 0
            correlations.append(correlation + 1e-10)
        log_radii = np.log(radii)
        log_correlations = np.log(correlations)
        valid_indices = np.isfinite(log_correlations) & np.isfinite(log_radii)
        if np.sum(valid_indices) < 3:
            return np.nan
        slope, _ = np.polyfit(log_radii[valid_indices], log_correlations[valid_indices], 1)
        return max(0, min(slope, 10))
    except:
        return np.nan

def convert_audio_to_wav(audio_file):
    """Convert audio file to WAV format using pydub."""
    if not PYDUB_AVAILABLE:
        # Fallback: only accept WAV files
        file_extension = audio_file.name.split('.')[-1].lower()
        if file_extension == 'wav':
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_file.write(audio_file.read())
            temp_file.close()
            return temp_file.name
        else:
            st.error("⚠️ Audio conversion not available. Please upload a WAV file.")
            return None
    
    try:
        file_extension = audio_file.name.split('.')[-1].lower()
        if file_extension == 'wav':
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_file.write(audio_file.read())
            temp_file.close()
            return temp_file.name
        
        if which("ffmpeg") is None:
            st.error("⚠️ ffmpeg not found. Cannot convert audio files.")
            st.info("Please install ffmpeg or upload a WAV file instead.")
            return None
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as temp_input:
            temp_input.write(audio_file.read())
            temp_input.flush()
            try:
                audio = AudioSegment.from_file(temp_input.name)
                temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                audio.export(temp_wav.name, format='wav')
                temp_wav.close()
                os.unlink(temp_input.name)
                return temp_wav.name
            except Exception as conv_error:
                if os.path.exists(temp_input.name):
                    os.unlink(temp_input.name)
                raise conv_error
    except Exception as e:
        st.error(f"Error converting audio file: {str(e)}")
        st.info("Please try uploading a WAV file instead, or install ffmpeg for audio conversion.")
        return None

def extract_parkinsons_features(audio_file):
    """Extract Parkinson's disease specific voice features"""
    try:
        wav_path = convert_audio_to_wav(audio_file)
        if wav_path is None:
            return {}, ["Failed to convert audio file"], None, None
        
        snd = parselmouth.Sound(wav_path)
        pitch = snd.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=500)
        f0_values = pitch.selected_array['frequency']
        f0_values = f0_values[f0_values > 0]
        point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)
        features = {}
        warnings_list = []
        num_points = parselmouth.praat.call(point_process, "Get number of points")
        if num_points < 3:
            warnings_list.append(f"Only {num_points} voiced segments found. Some features may be unreliable.")

        # Jitter and Shimmer features
        try:
            jitter_local = parselmouth.praat.call([point_process, pitch], "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            features["Jitter(%)"] = jitter_local * 100
            features["Jitter(Abs)"] = parselmouth.praat.call([point_process, pitch], "Get jitter (absolute)", 0, 0, 0.0001, 0.02, 1.3)
            features["Jitter:RAP"] = parselmouth.praat.call([point_process, pitch], "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
            features["Jitter:PPQ5"] = parselmouth.praat.call([point_process, pitch], "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
            features["Jitter:DDP"] = parselmouth.praat.call([point_process, pitch], "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
            features["Jitter_Method"] = "Praat"
        except:
            warnings_list.append("Praat jitter calculation failed, using mathematical fallback.")
            math_jitter = calculate_jitter_mathematical(f0_values)
            features.update(math_jitter)
            features["Jitter_Method"] = "Mathematical"

        try:
            features["Shimmer"] = parselmouth.praat.call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            features["Shimmer(dB)"] = parselmouth.praat.call([snd, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            features["Shimmer:APQ3"] = parselmouth.praat.call([snd, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            features["Shimmer:APQ5"] = parselmouth.praat.call([snd, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            features["Shimmer:APQ11"] = parselmouth.praat.call([snd, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            features["Shimmer:DDA"] = parselmouth.praat.call([snd, point_process], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        except Exception as e:
            warnings_list.append(f"Could not calculate shimmer features: {str(e)}")
            features.update({"Shimmer": np.nan, "Shimmer(dB)": np.nan, "Shimmer:APQ3": np.nan, "Shimmer:APQ5": np.nan, "Shimmer:APQ11": np.nan, "Shimmer:DDA": np.nan})

        try:
            harmonicity = snd.to_harmonicity_cc()
            hnr = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)
            features["HNR"] = hnr
            features["NHR"] = 1 / (10 ** (hnr / 10)) if hnr > -np.inf else np.nan
        except:
            features["HNR"] = np.nan
            features["NHR"] = np.nan
        
        try:
            if len(f0_values) > 50:
                features["RPDE"] = calculate_rpde(f0_values)
                features["DFA"] = calculate_dfa(f0_values)
                features["PPE"] = calculate_ppe(f0_values)
                features["Spread1"] = calculate_spread1(f0_values)
                features["Spread2"] = calculate_spread2(f0_values)
                features["D2"] = calculate_D2(f0_values)
            else:
                warnings_list.append("Insufficient data for nonlinear analysis features")
                features.update({"RPDE": np.nan, "DFA": np.nan, "PPE": np.nan, "Spread1": np.nan, "Spread2": np.nan, "D2": np.nan})
        except:
            warnings_list.append("Could not calculate nonlinear features.")
            features.update({"RPDE": np.nan, "DFA": np.nan, "PPE": np.nan, "Spread1": np.nan, "Spread2": np.nan, "D2": np.nan})
        
        os.unlink(wav_path)
        return features, warnings_list, f0_values, snd
    except Exception as e:
        return None, [f"Error processing audio file: {str(e)}"], None, None

def create_radar_chart(features):
    """Create a radar chart for voice features"""
    radar_features = ["Jitter(%)", "Shimmer", "HNR", "RPDE", "DFA", "PPE"]
    values, labels = [], []
    for feature in radar_features:
        if feature in features and not pd.isna(features[feature]):
            normalized = 0
            if feature == "Jitter(%)": normalized = min(features[feature] / 2.0, 1.0)
            elif feature == "Shimmer": normalized = min(features[feature] / 0.2, 1.0)
            elif feature == "HNR": normalized = max(0, min(features[feature] / 30.0, 1.0))
            else: normalized = min(features[feature] / 2.0, 1.0)
            values.append(normalized)
            labels.append(feature)
    if len(values) < 3: return None
    values += values[:1]
    labels += labels[:1]
    fig = go.Figure(go.Scatterpolar(r=values, theta=labels, fill='toself', name='Voice Features', line_color='rgb(66, 133, 244)'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=False, title="Voice Features Radar Chart (Normalized)")
    return fig

def prepare_features_for_prediction(features):
    """
    Prepare features for model prediction by mapping extracted features to expected model format.
    """
    expected_features = [
        'mdvp_fo_hz', 'mdvp_fhi_hz', 'mdvp_flo_hz', 'mdvp_jitter_%', 'mdvp_jitter_abs',
        'mdvp_rap', 'mdvp_ppq', 'jitter_ddp', 'mdvp_shimmer', 'mdvp_shimmer_db',
        'shimmer_apq3', 'shimmer_apq5', 'mdvp_apq', 'shimmer_dda', 'nhr', 'hnr',
        'rpde', 'dfa', 'spread1', 'spread2', 'd2', 'ppe'
    ]
    feature_mapping = {
        'Jitter(%)': 'mdvp_jitter_%', 'Jitter(Abs)': 'mdvp_jitter_abs', 'Jitter:RAP': 'mdvp_rap',
        'Jitter:PPQ5': 'mdvp_ppq', 'Jitter:DDP': 'jitter_ddp', 'Shimmer': 'mdvp_shimmer',
        'Shimmer(dB)': 'mdvp_shimmer_db', 'Shimmer:APQ3': 'shimmer_apq3', 'Shimmer:APQ5': 'shimmer_apq5',
        'Shimmer:APQ11': 'mdvp_apq', 'Shimmer:DDA': 'shimmer_dda', 'NHR': 'nhr', 'HNR': 'hnr',
        'RPDE': 'rpde', 'DFA': 'dfa', 'Spread1': 'spread1', 'Spread2': 'spread2', 'D2': 'd2', 'PPE': 'ppe'
    }
    
    f0_mean = 150.0
    if 'f0_values' in features and features['f0_values'] is not None:
        f0_vals = features['f0_values']
        if len(f0_vals) > 0:
            f0_mean = np.mean(f0_vals)
    
    feature_vector, missing_features = [], []
    for expected_feature in expected_features:
        value = None
        for extracted_feature, mapped_feature in feature_mapping.items():
            if mapped_feature == expected_feature and extracted_feature in features:
                value = features[extracted_feature]
                break
        
        if value is None:
            if expected_feature == 'mdvp_fo_hz': value = f0_mean
            elif expected_feature == 'mdvp_fhi_hz': value = f0_mean * 1.2
            elif expected_feature == 'mdvp_flo_hz': value = f0_mean * 0.8
            else: value = 0.0
            missing_features.append(expected_feature)
        
        if pd.isna(value):
            value = 0.0
            missing_features.append(expected_feature)
        
        feature_vector.append(float(value))
    
    X_df = pd.DataFrame([feature_vector], columns=expected_features)
    return X_df, missing_features

def make_parkinson_prediction(features):
    """Predict Parkinson's using the trained Random Forest model."""
    if not SKLEARN_AVAILABLE:
        st.warning("⚠️ scikit-learn not available. ML predictions disabled.")
        return None, [0.5, 0.5], 0.5, []
        
    if rf_model is None or rf_scaler is None:
        st.info('📝 Note: ML model not loaded. Predictions unavailable.')
        return None, [0.5, 0.5], 0.5, []
        
    X_df, missing = prepare_features_for_prediction(features)
    
    try:
        # Scale the features
        X_df_scaled = rf_scaler.fit_transform(X_df)
        pred = rf_model.predict(X_df_scaled)[0]
        proba = rf_model.predict_proba(X_df_scaled)[0]
        confidence = np.max(proba)
        return pred, proba, confidence, missing
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, [0.5, 0.5], 0.5, missing

# --- Gait Analysis Functions ---

@st.cache_resource
def load_pose_model():
    """Initializes the MediaPipe Pose model with suppressed warnings."""
    stderr_backup = sys.stderr
    sys.stderr = StringIO()
    try:
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    finally:
        sys.stderr = stderr_backup
    return mp_pose, mp_drawing, pose

mp_pose, mp_drawing, pose = load_pose_model()

LEFT_LEG_LANDMARKS = [23, 25, 27, 29, 31]
RIGHT_LEG_LANDMARKS = [24, 26, 28, 30, 32]

def process_video_frame(frame, pose_model):
    """Processes a single video frame for pose detection."""
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose_model.process(image)
    image.flags.writeable = True
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,66,230), thickness=2)
        )
    return image, results

def extract_gait_data(landmarks, frame_time):
    """Extracts gait-related data from MediaPipe landmarks."""
    left_vals = [round(landmarks[i].y, 4) for i in LEFT_LEG_LANDMARKS]
    right_vals = [round(landmarks[i].y, 4) for i in RIGHT_LEG_LANDMARKS]
    total_force_left = round(sum([landmarks[27].y, landmarks[29].y, landmarks[31].y]), 4)
    total_force_right = round(sum([landmarks[28].y, landmarks[30].y, landmarks[32].y]), 4)
    return {
        'Time': frame_time,
        'Left_Values': left_vals,
        'Right_Values': right_vals,
        'Total_Force_Left': total_force_left,
        'Total_Force_Right': total_force_right
    }

# --- Keystroke Dynamics Functions ---

def calculate_keystroke_features(df):
    """Calculate comprehensive keystroke dynamics features from a DataFrame."""
    try:
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
        df['relativeTime'] = pd.to_numeric(df['relativeTime'], errors='coerce')
        df = df.dropna(subset=['timestamp', 'relativeTime'])
        if len(df) < 4: return None, ["Insufficient data for analysis"], None, None
        
        keydowns = df[df['event'] == 'keydown'].reset_index(drop=True)
        keyups = df[df['event'] == 'keyup'].reset_index(drop=True)
        
        features, warnings, hold_times, flight_times = {}, [], [], []
        
        # Hold Times
        valid_keys = []
        for i in range(min(len(keydowns), len(keyups))):
            if keydowns.loc[i, 'key'] == keyups.loc[i, 'key']:
                hold_time = keyups.loc[i, 'timestamp'] - keydowns.loc[i, 'timestamp']
                if 0 < hold_time < 1000:
                    hold_times.append(hold_time)
                    valid_keys.append(keydowns.loc[i, 'key'])
        if hold_times:
            features.update({'Hold_Time_Mean': np.mean(hold_times), 'Hold_Time_Std': np.std(hold_times),
                             'Hold_Time_CV': np.std(hold_times) / np.mean(hold_times),
                             'Hold_Time_Min': np.min(hold_times), 'Hold_Time_Max': np.max(hold_times)})
        else: warnings.append("Could not calculate hold times")
        
        # Flight Times
        if len(keydowns) > 1:
            for i in range(1, len(keydowns)):
                flight_time = keydowns.loc[i, 'timestamp'] - keydowns.loc[i-1, 'timestamp']
                if 0 < flight_time < 5000: flight_times.append(flight_time)
        if flight_times:
            features.update({'Flight_Time_Mean': np.mean(flight_times), 'Flight_Time_Std': np.std(flight_times),
                             'Flight_Time_CV': np.std(flight_times) / np.mean(flight_times),
                             'Flight_Time_Min': np.min(flight_times), 'Flight_Time_Max': np.max(flight_times)})
        else: warnings.append("Could not calculate flight times")
        
        # Typing Speed and Pause Analysis
        if len(keydowns) > 0 and 'relativeTime' in df.columns:
            total_time = df['relativeTime'].max() / 1000.0
            if total_time > 0:
                features['Typing_Speed_CPS'] = len(keydowns) / total_time
                features['Typing_Speed_WPM'] = (len(keydowns) / 5) / (total_time / 60)
                features['Total_Typing_Time'] = total_time
        if flight_times:
            long_pauses = [ft for ft in flight_times if ft > 500]
            features['Long_Pauses_Count'] = len(long_pauses)
            features['Long_Pauses_Ratio'] = len(long_pauses) / len(flight_times)
            if long_pauses: features['Long_Pauses_Mean'] = np.mean(long_pauses)
        
        return features, warnings, hold_times, flight_times
    except Exception as e:
        return None, [f"Error calculating features: {str(e)}"], None, None

def create_keystroke_visualizations(hold_times, flight_times):
    """Create comprehensive visualizations for keystroke analysis"""
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=('Hold Times Distribution', 'Flight Times Distribution', 
                                        'Hold vs Flight Times', 'Keystroke Patterns Over Time'))
    if hold_times:
        fig.add_trace(go.Histogram(x=hold_times, name="Hold Times", nbinsx=20, marker_color='lightblue', opacity=0.7), row=1, col=1)
    if flight_times:
        fig.add_trace(go.Histogram(x=flight_times, name="Flight Times", nbinsx=20, marker_color='lightcoral', opacity=0.7), row=1, col=2)
        fig.add_trace(go.Scatter(x=list(range(len(flight_times))), y=flight_times, mode='lines+markers', name="Flight Time Sequence", line=dict(color='red', width=2)), row=2, col=2)
    if hold_times and flight_times:
        min_len = min(len(hold_times), len(flight_times))
        fig.add_trace(go.Scatter(x=hold_times[:min_len], y=flight_times[:min_len], mode='markers', name="Hold vs Flight", marker=dict(color='green', size=8, opacity=0.6)), row=2, col=1)
    fig.update_layout(height=800, title_text="Keystroke Dynamics Analysis")
    fig.update_xaxes(title_text="Hold Time (ms)", row=1, col=1)
    fig.update_xaxes(title_text="Flight Time (ms)", row=1, col=2)
    fig.update_xaxes(title_text="Hold Time (ms)", row=2, col=1)
    fig.update_xaxes(title_text="Keystroke Sequence", row=2, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    fig.update_yaxes(title_text="Flight Time (ms)", row=2, col=1)
    fig.update_yaxes(title_text="Flight Time (ms)", row=2, col=2)
    return fig

def assess_keystroke_patterns(features):
    """Assess keystroke patterns for potential Parkinson's indicators"""
    assessment = {"score": 0, "indicators": [], "recommendations": []}
    if not features: return assessment
    
    if features.get('Hold_Time_CV', 0) > 0.3:
        assessment["indicators"].append("High hold time variability (possible tremor)")
        assessment["score"] += 1
    if features.get('Flight_Time_CV', 0) > 0.4:
        assessment["indicators"].append("High flight time variability (possible motor control issues)")
        assessment["score"] += 1
    if features.get('Typing_Speed_CPS', 0) < 2.0:
        assessment["indicators"].append("Slow typing speed (possible bradykinesia)")
        assessment["score"] += 1
    if features.get('Long_Pauses_Ratio', 0) > 0.3:
        assessment["indicators"].append("Frequent long pauses (possible cognitive/motor planning issues)")
        assessment["score"] += 1
    if features.get('Hold_Time_Mean', 0) < 50:
        assessment["indicators"].append("Very short key presses (possible micrographia-like pattern)")
        assessment["score"] += 1
    
    if assessment["score"] == 0:
        assessment["recommendations"].append("Keystroke patterns appear normal")
    elif assessment["score"] <= 2:
        assessment["recommendations"].append("Minor variations detected - consider follow-up testing")
    else:
        assessment["recommendations"].append("Multiple indicators present - clinical evaluation recommended")
    return assessment


# --- Streamlit UI Components for Each Feature ---

def show_voice_analysis():
    # Custom CSS
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .feature-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .normal-range {
            color: #28a745;
            font-weight: bold;
        }
        .abnormal-range {
            color: #dc3545;
            font-weight: bold;
        }
        .warning-range {
            color: #ffc107;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🎤 Parkinson\'s Disease Voice Analysis</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📋 Analysis Options")
    st.sidebar.markdown("Upload an audio file to analyze voice features commonly used in Parkinson's disease detection research.")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'flac', 'm4a'],
        help="Upload a voice recording for analysis. WAV format is recommended."
    )
    
    if uploaded_file is not None:
        # Display audio player
        st.sidebar.audio(uploaded_file, format='audio/wav')
        
        # Analysis button
        if st.sidebar.button("🔬 Analyze Voice", type="primary"):
            with st.spinner("Analyzing voice features... This may take a moment."):
                features, warnings_list, f0_values, sound = extract_parkinsons_features(uploaded_file)
            
            if features is not None:
                # Main results
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.header("📊 Analysis Results")
                    
                    # Create tabs for different views
                    tab1, tab2, tab3 = st.tabs(["📈 Feature Overview", "📋 Detailed Results", "📊 Visualizations"])
                    
                    with tab1:
                        # Key metrics
                        metrics_cols = st.columns(4)
                        
                        key_features = ["HNR", "Shimmer", "Jitter(%)", "RPDE"]
                        for i, feature in enumerate(key_features):
                            if feature in features and not pd.isna(features[feature]):
                                status, color = get_feature_status(feature, features[feature])
                                metrics_cols[i].metric(
                                    label=feature,
                                    value=f"{features[feature]:.4f}",
                                    delta=status
                                )
                        
                        # Overall assessment
                        st.subheader("🎯 Overall Assessment")
                        available_features = sum(1 for k, v in features.items() 
                                               if not pd.isna(v) and isinstance(v, (int, float)) and k != "Jitter_Method")
                        total_features = len([k for k in features.keys() if k != "Jitter_Method"])
                        
                        if available_features >= 12:
                            st.success(f"✅ Comprehensive analysis completed ({available_features}/{total_features} features extracted)")
                        elif available_features >= 8:
                            st.warning(f"⚠️ Partial analysis completed ({available_features}/{total_features} features extracted)")
                        else:
                            st.error(f"❌ Limited analysis ({available_features}/{total_features} features extracted)")
                    
                    with tab2:
                        # Detailed feature table
                        st.subheader("📋 Complete Feature Analysis")
                        
                        feature_descriptions = {
                            "Jitter(%)": "Variation in fundamental frequency (%)",
                            "Jitter(Abs)": "Absolute variation in fundamental frequency",
                            "Jitter:RAP": "Relative Average Perturbation",
                            "Jitter:PPQ5": "5-point Period Perturbation Quotient",
                            "Jitter:DDP": "Average absolute difference of periods",
                            "Shimmer": "Amplitude variation",
                            "Shimmer(dB)": "Amplitude variation (dB)",
                            "Shimmer:APQ3": "3-point Amplitude Perturbation Quotient",
                            "Shimmer:APQ5": "5-point Amplitude Perturbation Quotient",
                            "Shimmer:APQ11": "11-point Amplitude Perturbation Quotient",
                            "Shimmer:DDA": "Average absolute amplitude differences",
                            "HNR": "Harmonics-to-Noise Ratio (dB)",
                            "NHR": "Noise-to-Harmonics Ratio",
                            "RPDE": "Recurrence Period Density Entropy",
                            "DFA": "Detrended Fluctuation Analysis",
                            "PPE": "Pitch Period Entropy"
                        }
                        
                        # Create detailed results table
                        results_data = []
                        for feature_name, value in features.items():
                            # Skip internal method indicators
                            if feature_name == "Jitter_Method":
                                continue
                                
                            status, color = get_feature_status(feature_name, value)
                            
                            # Format value appropriately
                            if pd.isna(value):
                                formatted_value = "N/A"
                            elif isinstance(value, (int, float)):
                                formatted_value = f"{value:.6f}"
                            else:
                                formatted_value = str(value)
                            
                            results_data.append({
                                "Feature": feature_name,
                                "Value": formatted_value,
                                "Status": status,
                                "Description": feature_descriptions.get(feature_name, "Voice feature")
                            })
                        
                        results_df = pd.DataFrame(results_data)
                        st.dataframe(results_df, use_container_width=True)
                    
                    with tab3:
                        # Visualizations
                        st.subheader("📈 Voice Feature Visualizations")
                        
                        # Radar chart
                        radar_fig = create_radar_chart(features)
                        if radar_fig:
                            st.plotly_chart(radar_fig, use_container_width=True)
                        
                        # Feature distribution
                        if f0_values is not None and len(f0_values) > 0:
                            fig_f0 = px.line(
                                x=range(len(f0_values)), 
                                y=f0_values,
                                title="Fundamental Frequency (F0) Over Time",
                                labels={"x": "Time Frames", "y": "F0 (Hz)"}
                            )
                            st.plotly_chart(fig_f0, use_container_width=True)
                        
                        # Feature comparison bars
                        available_features_viz = {k: v for k, v in features.items() 
                                            if not pd.isna(v) and isinstance(v, (int, float)) and k != "Jitter_Method"}
                        if len(available_features_viz) > 0:
                            fig_bar = px.bar(
                                x=list(available_features_viz.keys()),
                                y=list(available_features_viz.values()),
                                title="Extracted Voice Features",
                                labels={"x": "Features", "y": "Values"}
                            )
                            fig_bar.update_layout(xaxis_tickangle=45)
                            st.plotly_chart(fig_bar, use_container_width=True)
                
                with col2:
                    st.header("ℹ️ Information")
                    
                    # Warnings
                    if warnings_list:
                        st.subheader("⚠️ Analysis Warnings")
                        for warning in warnings_list:
                            st.warning(warning)
                    
                    # Feature info
                    st.subheader("📖 About the Features")
                    st.markdown("""
                    **Jitter Features**: Measure variation in vocal pitch
                    - Higher values may indicate voice disorders
                    
                    **Shimmer Features**: Measure variation in vocal amplitude  
                    - Higher values may indicate voice disorders
                    
                    **HNR/NHR**: Voice quality measures
                    - HNR: Higher is better (less noise)
                    - NHR: Lower is better (less noise)
                    
                    **Nonlinear Features**: Complex voice dynamics
                    - RPDE, DFA, PPE: Measure voice complexity and stability
                    """)
                    
                    # Mathematical formulas expander
                    with st.expander("🔢 Mathematical Formulas"):
                        st.markdown("""
                        **Jitter Calculations:**
                        
                        Given fundamental frequency values F₀, convert to periods: T = 1/F₀
                        
                        • **Jitter(%)**: `100 × (Σ|Tᵢ₊₁ - Tᵢ|/(N-1)) / (ΣTᵢ/N)`
                        
                        • **Jitter(Abs)**: `Σ|Tᵢ₊₁ - Tᵢ|/(N-1)` (in seconds)
                        
                        • **RAP**: `(Σ|Tᵢ - (Tᵢ₋₁+Tᵢ+Tᵢ₊₁)/3|/(N-2)) / T̄`
                        
                        • **PPQ5**: `(Σ|Tᵢ - T̄₅ᵢ|/(N-4)) / T̄`
                          where T̄₅ᵢ is 5-point local average
                        
                        • **DDP**: `(Σ|(Tᵢ₊₁-Tᵢ) - (Tᵢ-Tᵢ₋₁)|/(N-2)) / T̄`
                        
                        Where:
                        - N = number of periods
                        - T̄ = mean period duration
                        - |x| = absolute value
                        """)
                    
                    # Show calculation method
                    if 'features' in locals() and features is not None and 'Jitter_Method' in features:
                        method = features['Jitter_Method']
                        if method == "Praat":
                            st.info("🔬 Jitter calculated using Praat algorithms")
                        else:
                            st.info("🧮 Jitter calculated using mathematical formulas")
                    
                    # ML Prediction Section
                    st.subheader("🤖 AI Prediction")
                    prediction, prediction_proba, confidence, missing_features = make_parkinson_prediction(features)
                    if prediction is not None:
                        if prediction == 1:
                            st.error(f"⚠️ **Positive for Parkinson's Disease Indicators** - Probability: {prediction_proba[1]:.1%}")
                        else:
                            st.success(f"✅ **No Parkinson's Disease Indicators Detected** - Probability: {prediction_proba[0]:.1%}")
                        
                        st.write(f"**Model Confidence:** {confidence:.1%}")
                        st.progress(confidence)
                        
                        # Probability visualization
                        prob_df = pd.DataFrame({'Class': ['Healthy', 'Parkinson\'s'], 'Probability': [prediction_proba[0], prediction_proba[1]]})
                        fig_prob = px.bar(prob_df, x='Class', y='Probability', color='Probability', 
                                         color_continuous_scale='RdYlGn_r', title="Prediction Probabilities")
                        fig_prob.update_layout(showlegend=False, height=300)
                        st.plotly_chart(fig_prob, use_container_width=True)
                        
                        if missing_features:
                            st.warning(f"⚠️ Note: {len(missing_features)} features were missing or estimated.")
                            with st.expander("View missing/estimated features"):
                                st.write(", ".join(missing_features))
                    else:
                        st.info("📝 ML prediction not available (model not loaded or scikit-learn not installed)")
                    
                    # Download results
                    st.subheader("💾 Download Results")
                    csv_buffer = StringIO()
                    results_df.to_csv(csv_buffer, index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv_buffer.getvalue(),
                        file_name=f"voice_analysis_{uploaded_file.name}.csv",
                        mime="text/csv"
                    )
                
                # Disclaimer
                st.markdown("---")
                st.markdown("""
                **⚠️ Disclaimer**: This tool is for research and educational purposes only. 
                It should not be used for medical diagnosis. Always consult healthcare professionals 
                for medical concerns.
                """)
            
            else:
                st.error("❌ Failed to analyze the audio file. Please check the file format and try again.")
                if warnings_list:
                    for warning in warnings_list:
                        st.error(warning)
    
    else:
        # Instructions when no file is uploaded
        st.info("👆 Please upload an audio file using the sidebar to begin analysis.")
        
        # Feature information
        st.header("🔬 About Parkinson's Voice Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Features Analyzed")
            st.markdown("""
            - **Jitter (5 measures)**: Frequency variation
            - **Shimmer (6 measures)**: Amplitude variation  
            - **HNR/NHR**: Voice quality ratios
            - **RPDE**: Recurrence analysis
            - **DFA**: Fractal scaling
            - **PPE**: Pitch entropy
            """)
        
        with col2:
            st.subheader("🎯 Research Applications")
            st.markdown("""
            - Voice disorder detection
            - Parkinson's disease research
            - Speech therapy assessment
            - Voice quality monitoring
            - Clinical voice analysis
            """)

def get_feature_status(feature_name, value):
    """Determine if feature value is normal, abnormal, or excellent"""
    normal_ranges = {
        "Jitter(%)": (0.1, 0.5),
        "Jitter(Abs)": (0.00002, 0.00005),
        "Jitter:RAP": (0.1, 0.5),
        "Jitter:PPQ5": (0.1, 0.5),
        "Jitter:DDP": (0.3, 1.5),
        "Shimmer": (0.02, 0.05),
        "Shimmer(dB)": (0.2, 0.5),
        "Shimmer:APQ3": (0.01, 0.04),
        "Shimmer:APQ5": (0.02, 0.05),
        "Shimmer:APQ11": (0.03, 0.08),
        "Shimmer:DDA": (0.03, 0.12),
        "HNR": (20, 30),
        "NHR": (0.01, 0.05),
        "RPDE": (0.4, 0.7),
        "DFA": (0.5, 0.9),
        "PPE": (0.1, 0.3)
    }
    
    if pd.isna(value):
        return "N/A", "gray"
    
    if feature_name not in normal_ranges:
        return "Measured", "blue"
    
    min_val, max_val = normal_ranges[feature_name]
    
    if min_val <= value <= max_val:
        return "Normal", "green"
    elif feature_name == "HNR" and value > max_val:
        return "Excellent", "darkgreen"
    elif feature_name == "NHR" and value < min_val:
        return "Excellent", "darkgreen"
    else:
        return "Abnormal", "red"

def show_gait_analysis():
    st.title("🚶 Video-Based Gait Analysis")
    st.markdown("---")
    st.markdown("📹 **Upload a video of a person walking for automatic gait analysis.**")
    
    st.sidebar.header("📊 Controls")
    confidence_threshold = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.5, 0.1)
    tracking_threshold = st.sidebar.slider("Tracking Confidence", 0.1, 1.0, 0.5, 0.1)
    
    uploaded_file = st.file_uploader(
        "Choose a gait video file",
        type=['mp4', 'avi', 'mov', 'mkv', 'wmv'],
        help="Upload a video file showing body movement for gait analysis"
    )

    if 'gait_data' not in st.session_state:
        st.session_state.gait_data = []
    
    if uploaded_file is not None:
        analyze_button = st.button("🎥 Analyze Video", type="primary")
        if analyze_button:
            st.session_state.gait_data = []
            
            pose_dynamic = mp_pose.Pose(min_detection_confidence=confidence_threshold, min_tracking_confidence=tracking_threshold)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_video_path = tmp_file.name
            
            try:
                cap = cv2.VideoCapture(temp_video_path)
                if cap.isOpened():
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count, detected_poses = 0, 0
                    
                    progress_bar = st.progress(0)
                    video_placeholder = st.empty()
                    
                    while True:
                        ret, frame = cap.read()
                        if not ret: break
                        frame_count += 1
                        frame_time = round(frame_count / fps, 3) if fps > 0 else frame_count
                        progress_bar.progress(frame_count / total_frames)
                        
                        annotated_frame, results = process_video_frame(frame, pose_dynamic)
                        video_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)
                        
                        if results.pose_landmarks:
                            gait_data = extract_gait_data(results.pose_landmarks.landmark, frame_time)
                            st.session_state.gait_data.append(gait_data)
                            detected_poses += 1
                        time.sleep(0.03)
                    cap.release()
                    pose_dynamic.close()
                    st.success(f"✅ Analysis completed! {detected_poses} poses detected from {frame_count} frames.")
                    st.balloons()
                else:
                    st.error("❌ Could not open video file.")
            except Exception as e:
                st.error(f"❌ Error processing video: {str(e)}")
            finally:
                if os.path.exists(temp_video_path): os.unlink(temp_video_path)

    if st.session_state.gait_data:
        st.subheader("📊 Analysis Results")
        total_data_points = len(st.session_state.gait_data)
        if total_data_points > 0:
            avg_left_force = sum(d['Total_Force_Left'] for d in st.session_state.gait_data) / total_data_points
            avg_right_force = sum(d['Total_Force_Right'] for d in st.session_state.gait_data) / total_data_points
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Total Data Points", total_data_points)
            with col2: st.metric("Avg Left Force", f"{avg_left_force:.4f}")
            with col3: st.metric("Avg Right Force", f"{avg_right_force:.4f}")

        df_display = pd.DataFrame([
            {'Time': data['Time'], 
             'L1': data['Left_Values'][0] if len(data['Left_Values']) > 0 else 0, 
             'L2': data['Left_Values'][1] if len(data['Left_Values']) > 1 else 0,
             'R1': data['Right_Values'][0] if len(data['Right_Values']) > 0 else 0, 
             'R2': data['Right_Values'][1] if len(data['Right_Values']) > 1 else 0} 
            for data in st.session_state.gait_data
        ])
        st.subheader("📋 Data Preview")
        st.dataframe(df_display.head(10), use_container_width=True)
        if len(df_display) > 10: 
            st.info(f"Showing first 10 rows. Complete dataset has {len(df_display)} rows.")

        # Create complete gait data for CSV export
        csv_data = []
        for data in st.session_state.gait_data:
            row = {'Time': data['Time']}
            # Add left leg values
            for i, val in enumerate(data['Left_Values'][:8]):
                row[f'L{i+1}'] = val
            # Add right leg values  
            for i, val in enumerate(data['Right_Values'][:8]):
                row[f'R{i+1}'] = val
            row['Total_Force_Left'] = data['Total_Force_Left']
            row['Total_Force_Right'] = data['Total_Force_Right']
            csv_data.append(row)

        csv_buffer = StringIO()
        pd.DataFrame(csv_data).to_csv(csv_buffer, index=False)
        st.download_button(
            label="📅 Download Complete CSV Dataset",
            data=csv_buffer.getvalue(),
            file_name=f"gait_analysis_{int(time.time())}.csv",
            mime="text/csv"
        )
    else:
        st.info("📄 Please upload a video file to begin analysis.")

def show_keystroke_analysis():
    st.title("⌨️ Keystroke Dynamics for Parkinson's Detection")
    st.markdown("---")
    st.markdown("""
    **Advanced keystroke pattern analysis for neurological assessment**
    
    Type the sentence below to capture detailed keystroke timing data for Parkinson's disease research.
    
    > **Target Sentence: "The quick brown fox jumps over the lazy dog"**
    """)
    
    html_code = """
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        textarea { width: 100%; height: 150px; font-size: 18px; padding: 10px; border: 2px solid #ddd; border-radius: 5px; font-family: 'Courier New', monospace; line-height: 1.5; }
        .controls { margin: 10px 0; display: flex; gap: 10px; align-items: center; }
        button { background-color: #ff4b4b; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; font-size: 14px; }
        button:hover { background-color: #ff6b6b; }
        button:disabled { background-color: #ccc; cursor: not-allowed; }
        .status { margin-left: 10px; font-weight: bold; }
        .target-text { background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 10px; font-family: 'Courier New', monospace; }
    </style>
    </head>
    <body>
    <div class="target-text">
        <strong>Target:</strong> The quick brown fox jumps over the lazy dog
    </div>
    <textarea id="typingArea" placeholder="Start typing the target sentence here..."></textarea>
    <div class="controls">
        <button onclick="downloadCSV()" id="downloadBtn">📥 Download Raw Data</button>
        <button onclick="clearData()" id="clearBtn">🗑️ Clear & Restart</button>
        <span class="status" id="status">Ready to type...</span>
    </div>
    <script>
    let keyData = [];
    let startTime = null;
    const typingArea = document.getElementById('typingArea');
    const status = document.getElementById('status');
    typingArea.addEventListener('keydown', function (e) {
        if (!startTime) { startTime = performance.now(); }
        const timestamp = performance.now();
        keyData.push({ key: e.key, keyCode: e.keyCode, event: "keydown", timestamp: timestamp, relativeTime: timestamp - startTime });
        updateStatus();
    });
    typingArea.addEventListener('keyup', function (e) {
        const timestamp = performance.now();
        keyData.push({ key: e.key, keyCode: e.keyCode, event: "keyup", timestamp: timestamp, relativeTime: startTime ? timestamp - startTime : 0 });
        updateStatus();
    });
    typingArea.addEventListener('input', function (e) { updateStatus(); });
    function updateStatus() {
        const textLength = typingArea.value.length;
        const targetLength = "The quick brown fox jumps over the lazy dog".length;
        const keyPresses = keyData.filter(d => d.event === 'keydown').length;
        if (textLength === 0) { status.textContent = "Ready to type..."; }
        else if (textLength < targetLength) { status.textContent = `Typing... (${textLength}/${targetLength} characters, ${keyPresses} keystrokes)`; }
        else { status.textContent = `Complete! (${keyPresses} keystrokes captured)`; }
    }
    function getCSV() {
        if (keyData.length === 0) { alert('No keystroke data captured yet.'); return null; }
        let csv = "key,keyCode,event,timestamp,relativeTime\\n";
        csv += keyData.map(d => `"${d.key}",${d.keyCode},${d.event},${d.timestamp},${d.relativeTime}`).join("\\n");
        return csv;
    }
    function downloadCSV() {
        const csv = getCSV();
        if (!csv) return;
        const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, '-');
        const filename = `keystroke_data_${timestamp}.csv`;
        const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.style.display = 'none';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        status.textContent = `Downloaded: ${filename}`;
    }
    function clearData() {
        keyData = [];
        startTime = null;
        typingArea.value = '';
        status.textContent = "Ready to type...";
    }
    window.downloadCSV = downloadCSV;
    window.getCSV = getCSV;
    window.clearData = clearData;
    </script>
    </body>
    </html>
    """
    components.html(html_code, height=400)
    
    st.header("📊 Keystroke Analysis")
    uploaded_file = st.file_uploader("📁 Upload Keystroke Data CSV", type="csv")
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            if df.empty or 'timestamp' not in df.columns:
                st.error("Invalid file format.")
            else:
                st.success(f"✅ File uploaded successfully! ({len(df)} events)")
                with st.spinner("🔍 Analyzing keystroke patterns..."):
                    features, warnings_list, hold_times, flight_times = calculate_keystroke_features(df)
                if features:
                    tab1, tab2, tab3 = st.tabs(["📊 Key Metrics", "📈 Visualizations", "🎯 Assessment"])
                    with tab1:
                        st.subheader("🎯 Key Keystroke Metrics")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            if 'Hold_Time_Mean' in features: st.metric("Avg Hold Time", f"{features['Hold_Time_Mean']:.1f} ms")
                            if 'Flight_Time_Mean' in features: st.metric("Avg Flight Time", f"{features['Flight_Time_Mean']:.1f} ms")
                        with col2:
                            if 'Hold_Time_CV' in features: st.metric("Hold Time Variability", f"{features['Hold_Time_CV']:.3f}")
                            if 'Flight_Time_CV' in features: st.metric("Flight Time Variability", f"{features['Flight_Time_CV']:.3f}")
                        with col3:
                            if 'Typing_Speed_CPS' in features: st.metric("Typing Speed", f"{features['Typing_Speed_CPS']:.1f} chars/sec")
                            if 'Typing_Speed_WPM' in features: st.metric("Words Per Minute", f"{features['Typing_Speed_WPM']:.0f} WPM")
                        with col4:
                            if 'Long_Pauses_Count' in features: st.metric("Long Pauses", f"{features['Long_Pauses_Count']}")
                            if 'Total_Typing_Time' in features: st.metric("Total Time", f"{features['Total_Typing_Time']:.1f} sec")
                    with tab2:
                        st.subheader("📈 Keystroke Pattern Visualizations")
                        if hold_times or flight_times:
                            viz_fig = create_keystroke_visualizations(hold_times, flight_times)
                            st.plotly_chart(viz_fig, use_container_width=True)
                        else: st.warning("Insufficient data for visualizations")
                    with tab3:
                        st.subheader("🎯 Parkinson's Indicator Assessment")
                        assessment = assess_keystroke_patterns(features)
                        score_color = "green" if assessment["score"] <= 1 else "orange" if assessment["score"] <= 3 else "red"
                        st.markdown(f"**Risk Score:** <span style='color: {score_color}; font-size: 24px; font-weight: bold;'>{assessment['score']}/5</span>", unsafe_allow_html=True)
                        if assessment["indicators"]:
                            st.subheader("🔍 Detected Indicators")
                            for indicator in assessment["indicators"]: st.warning(f"• {indicator}")
                        st.subheader("💡 Recommendations")
                        for rec in assessment["recommendations"]: st.info(f"• {rec}")
                else:
                    st.error("❌ Could not analyze keystroke data.")
                    for warning in warnings_list: st.error(f"• {warning}")
        except Exception as e: st.error(f"❌ Error processing file: {str(e)}")
    else: st.info("👆 Please upload a keystroke CSV file to begin analysis.")


# --- Main Application Logic ---

def main():
    st.set_page_config(
        page_title="Parkinson's Multi-Modal Analysis",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.sidebar.title("🧠 Parkinson's Analysis Tools")
    tool_selection = st.sidebar.selectbox(
        "Select an Analysis Tool",
        ("Voice Analysis", "Gait Analysis", "Keystroke Dynamics")
    )

    if tool_selection == "Voice Analysis":
        show_voice_analysis()
    elif tool_selection == "Gait Analysis":
        show_gait_analysis()
    elif tool_selection == "Keystroke Dynamics":
        show_keystroke_analysis()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **⚠️ Medical Disclaimer:** This tool is for research and educational purposes only. It should not be used as a sole method for medical diagnosis. Always consult qualified healthcare professionals for medical concerns.
    """)

if __name__ == "__main__":
    main()

