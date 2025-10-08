import sys

# --- Paths ---
# Directory with many .wav files of diverse dialects
TRAIN_DIR = sys.argv[1]
# A single, clean .wav file from the target dialect for reference
REFERENCE_WAV = sys.argv[2]
# The .wav file from another dialect that you want to convert
SOURCE_WAV = sys.argv[3]


# Where to save the final converted audio file
OUTPUT_WAV = "./converted_output.wav"

# --- Model Parameters ---
# Number of principal components to model the articulation space
N_COMPONENTS = 8
# Device to run the PyTorch models on
DEVICE = "cuda:0"

# ===================================================================
# Part 2: Imports and Model Loading
# ===================================================================
import os
import glob
import numpy as np
import soundfile as sf
from sparc import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import librosa, tempfile

print("Loading the SPARC synthesis model...")
coder = load_model("en+", device=DEVICE)
print("Model loaded successfully.")


def process_and_encode(path, coder_instance, top_db=25):
    """
    Loads a WAV, trims silence, saves to a temporary file,
    and then encodes it using the provided coder instance.
    This is necessary because the encoder requires a file path.
    """
    try:
        # Load and resample to 16kHz, expected by the SPARC model
        y, sr = librosa.load(path, sr=16000)
        # Trim leading/trailing silence
        y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)

        # If audio is too short after trimming, it's not useful
        if len(y_trimmed) < 400: # ~25ms minimum
            return None

        # Create a named temporary file that is automatically deleted when closed.
        # The .wav suffix is important for some encoders.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_wav_file:
            # Save the trimmed audio data into the temporary file
            sf.write(temp_wav_file.name, y_trimmed, 16000)
            # Now, encode from the path of the temporary file
            params = coder_instance.encode(temp_wav_file.name)
            return params
    except Exception as e:
        print(f"  Error processing file {path}: {e}")
        return None

# ===================================================================
# Part 3: Training on the Target Dialect
# (Builds the PCA and F0 models)
# ===================================================================
print(f"\n--- Starting Training Phase on files in: {TRAIN_DIR} ---")

# Find all .wav files in the training directory
wav_files = glob.glob(os.path.join(TRAIN_DIR, '**', '*.wav'), recursive=True)[:10]
if not wav_files:
    raise FileNotFoundError(f"No .wav files found in {TRAIN_DIR}")

print(f"Found {len(wav_files)} audio files for training.")

# Collect data from all files
all_ema_data = []

for i, filepath in enumerate(wav_files):
    print(f"  Processing file {i+1}/{len(wav_files)}: {os.path.basename(filepath)}")
    try:
        params = process_and_encode(filepath, coder)
        # Append articulation (EMA) data
        if 'ema' in params:
            all_ema_data.append(params['ema'])
    except Exception as e:
        print(f"  Warning: Could not process {filepath}. Error: {e}")

# --- Train the PCA Model ---
print("\nTraining PCA model on aggregated EMA data...")
combined_ema = np.vstack(all_ema_data)
pca_scaler = StandardScaler()
scaled_ema = pca_scaler.fit_transform(combined_ema)

pca_model = PCA(n_components=N_COMPONENTS)
pca_model.fit(scaled_ema)

explained_variance = np.sum(pca_model.explained_variance_ratio_)
print(f"PCA model trained. Explained variance for {N_COMPONENTS} components: {explained_variance:.4f}")


print("--- Training Complete ---")


# ===================================================================
# Part 4: Analysis of the Reference Utterance
# (Finds the target articulatory posture)
# ===================================================================
print(f"\n--- Analyzing Reference File: {os.path.basename(REFERENCE_WAV)} ---")

# Encode the reference file to get its EMA
ref_params = process_and_encode(REFERENCE_WAV, coder)
if 'ema' not in ref_params:
    raise ValueError("Could not extract EMA data from the reference WAV file.")

# Transform the reference EMA into the trained PCA space
ref_scaled_ema = pca_scaler.transform(ref_params['ema'])
ref_pca_trajectories = pca_model.transform(ref_scaled_ema)

# The target means of the reference utterance in PCA space
target_pca_mean = np.mean(ref_pca_trajectories, axis=0)
target_pca_std = np.std(ref_pca_trajectories, axis=0)
print(f"Derived target articulation mean vector (shape: {target_pca_mean.shape})")
print("--- Reference Analysis Complete ---")


# ===================================================================
# Part 5: Conversion of the Source Utterance
# ===================================================================
print(f"\n--- Converting Source File: {os.path.basename(SOURCE_WAV)} ---")

# Encode the source file to get all its parameters
source_params = process_and_encode(SOURCE_WAV, coder)


# Create a copy of the parameters that we will modify for synthesis
params_for_synthesis = source_params.copy()

# --- 2. Articulation (EMA) Conversion via Mean Shift in PCA Space ---
print("Shifting source articulation to match reference mean...")

# Transform source EMA to the trained PCA space
source_scaled_ema = pca_scaler.transform(source_params['ema'])
source_pca_trajectories = pca_model.transform(source_scaled_ema)

# Calculate the source's own mean articulation
source_pca_mean = np.mean(source_pca_trajectories, axis=0)
source_pca_std = np.std(source_pca_trajectories, axis=0)
print("source means:", source_pca_mean, source_pca_std)
print("target means:", target_pca_mean, target_pca_std)

# The core conversion: transform means to match target
modified_pca_trajectories = (source_pca_trajectories - source_pca_mean) + target_pca_mean

# or transform both mean and std
centered_source = source_pca_trajectories - source_pca_mean
# 2. Scale the centered trajectories to have a standard deviation of 1.
normalized_source = centered_source / source_pca_std
# 3. Rescale and shift the normalized trajectories to match the target's stats.
modified_pca_trajectories = (normalized_source * target_pca_std) + target_pca_mean

print(np.mean(modified_pca_trajectories, axis=0))
CONVERSION_STRENGTH = 1.5
modified_pca_trajectories = (
    (1 - CONVERSION_STRENGTH) * source_pca_trajectories +
    CONVERSION_STRENGTH *  modified_pca_trajectories
)
# Transform the modified trajectories back to the EMA space
reconstructed_scaled_ema = pca_model.inverse_transform(modified_pca_trajectories)
reconstructed_ema = pca_scaler.inverse_transform(reconstructed_scaled_ema)
params_for_synthesis['ema'] = reconstructed_ema.astype(np.float32)

# --- 3. Synthesize and Save Audio ---
print("\nSynthesizing converted audio...")

wav_out = coder.decode(**params_for_synthesis)

sf.write(OUTPUT_WAV, wav_out, 16000)
print("desired dialect")
os.system("play "+REFERENCE_WAV)
print("before conversion:")
os.system("play "+SOURCE_WAV)
print("after conversion:")
os.system("play "+OUTPUT_WAV)
print(f"--- Conversion Complete! File saved to: {OUTPUT_WAV} ---")
