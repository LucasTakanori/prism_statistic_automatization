# Audio Embeddings Implementation Guide

To break the current performance ceiling ($R^2 \approx 0.30$), the next logical step is to utilize **Deep Learning Audio Embeddings** (e.g., Wav2Vec 2.0, HuBERT). These models capture rich phonetic and paralinguistic information that simple acoustic features (jitter, shimmer) miss.

## 1. Environment Setup
You will need to install PyTorch and Hugging Face Transformers.
```bash
uv pip install torch torchaudio transformers librosa
```

## 2. Embedding Extraction Pipeline
Create a new script (e.g., `extract_embeddings.py`) to process your `.wav` files.

### Recommended Model
*   **Wav2Vec 2.0** (`facebook/wav2vec2-base-960h` or `facebook/wav2vec2-large-xlsr-53` for multilingual).
*   **HuBERT** (`facebook/hubert-large-ls960-ft`).
*   **Unispeech-SAT** (Good for speaker-related tasks).

### Code Snippet (Template)
```python
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import pandas as pd
import os
import glob

# 1. Load Pre-trained Model
model_name = "facebook/wav2vec2-base-960h"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2Model.from_pretrained(model_name)
model.eval()

# 2. Loop through Audio
embeddings_list = []
audio_dir = "/path/to/wavs" # UPDATE THIS

for filepath in glob.glob(os.path.join(audio_dir, "*.wav")):
    # Load audio
    waveform, sample_rate = torchaudio.load(filepath)
    
    # Resample to 16kHz if necessary (Wav2Vec2 requires 16k)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
        
    # Process
    input_values = processor(waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=16000).input_values
    
    with torch.no_grad():
        outputs = model(input_values)
        
    # Extract Hidden States
    # outputs.last_hidden_state shape: (1, seq_len, 768)
    # We need a fixed-size vector per file.
    
    # Strategy A: Mean Pooling (Average over time)
    embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze().numpy()
    
    # Strategy B: Max Pooling, or use specific layer
    
    # Store
    filename = os.path.basename(filepath)
    # Parse subject/task from filename similar to current data loader
    # e.g. PZ001_phonationA.wav
    
    record = {"source_file": filename}
    # Add embedding features (emb_0, emb_1, ... emb_767)
    for i, val in enumerate(embedding):
        record[f"emb_{i}"] = val
        
    embeddings_list.append(record)

# 3. Save to CSV
df_emb = pd.DataFrame(embeddings_list)
df_emb.to_csv("wav2vec_features.csv", index=False)
```

## 3. Integration with ML Pipeline

Once you have `wav2vec_features.csv`:

1.  **Update `data_loader.py`**:
    *   Change the default `features_path` to your new CSV.
    *   The existing logic (Pivoting, Stratified Split) should work **as is** if you inspect `source_file` to extract `subject` and `task` correctly.
    *   *Note*: The dimensionality will be huge (768 dims * number_of_tasks_per_subject).
    
2.  **Dimensionality Reduction**:
    *   Since you will have ~5000+ features per subject (if you pivot 768 dims x 6 tasks), you **MUST** use dimensionality reduction.
    *   **PCA**: Apply Principal Component Analysis before training.
    *   **Feature Selection**: Use the existing `SelectKBest` with `k=50` or `k=100`.

3.  **Training**:
    *   Run the optimization script again. 
    *   **SVR** and **Ridge** are likely to perform best on high-dimensional embedding data.

## 4. Alternate Approach: Fine-Tuning
Instead of extracting static features, you can fine-tune the Wav2Vec2 model end-to-end with a Regression Head on top (predicting ALSFRS-R directly). This requires:
*   A custom `Dataset` class in PyTorch.
*   A training loop with GPU support.
*   More compute resources.
*   Likely yields the highest possible performance ($R^2 > 0.60$).
