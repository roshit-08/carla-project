Implementation Plan: 1D-CNN Model for Real-Time Harsh Driving Detection
We will implement a high-accuracy 1D-CNN model to detect harsh driving events in real-time, resolving the poor training performance of the previous CNN-BiLSTM and the real-time lookahead latency issues associated with bidirectional recurrent networks.

User Review Required
IMPORTANT

Architecture Decision: We will use a pure 1D-CNN model. It processes the sequence input in parallel, has a fixed temporal receptive field, and operates with zero causal lookahead latency, making it ideal for real-time manual control. We will also implement a 3D-to-2D flattening sampler to balance the sequence training set and a sequence-wide StandardScaler to ensure numerical stability.

Proposed Changes
We will group our work into the following phases:

Phase 1: CNN Model Training Notebook
[NEW] 
harsh_event_detection_cnn.ipynb
Create a new Jupyter Notebook that:

Loads the raw CSV dataset using load_row_labeled_data.
Applies the same rolling median denoising and targeted feature engineering (feature_cols_stream with 30 columns).
Segments the continuous sequences into sliding windows (window_size = 25, step_size = 5).
Performs a file-based split using GroupShuffleSplit (referencing the baseline manifest to maintain comparable test sets).
Standardizes sequence features: Fits a StandardScaler on the training set (flattened across timesteps) and standardizes train/val/test splits.
Balances classes: Temporarily flattens the train sequence tensor from [B, 25, 30] to [B, 750], applies RandomOverSampler, and reshapes it back to [B_bal, 25, 30].
Defines the 1D-CNN Model:
4 Convolutional blocks (1D Conv + BatchNorm + ReLU + Dropout).
Global average pooling to collapse time.
Fully connected classification layers.
Trains the model using AdamW, Cosine Annealing learning rate scheduler, and early stopping on validation Macro F1.
Saves the complete model bundle (PyTorch weights, label encoder, scaler, and configuration) to artifacts/harsh_event_cnn_bundle.pth.
Phase 2: Core Detector Integration
[MODIFY] 
harsh_event_model.py
Add a PyTorch-based sequence detector class: RealtimeCNNHarshEventDetector.
Implement feature construction for the stream: given incoming IMU packets, it maintains a buffer, computes the rolling features (jerk, magnitude, std roll, etc.) for feature_cols_stream on the fly, scales the sequence using the saved StandardScaler, and runs inference using the trained 1D-CNN PyTorch model.
Support the same temporal smoothing (consecutive_hits) and heuristic fallbacks.
Phase 3: Real-Time CARLA Run Integration
[NEW] 
run_carla_realtime_cnn.py
Create a script to run real-time inference using the CNN detector, passing flags like --invert-gyro-z and --invert-acc-y to correct coordinate handedness issues. This script will mirror the original script but import the new RealtimeCNNHarshEventDetector and run PyTorch-based inference.

Verification Plan
Automated Offline Test
Execute the training notebook cells and inspect the final test set classification report.
Ensure test Macro F1-score is high ($> 80%$) and confusion matrix shows good turn and lane change recall.
Manual Real-Time Verification
Run the mock stream script /home/pw26_rr_09/pw26_rr_07/baseModel/temporary/run_realtime_demo.py updated for the CNN model to verify that streaming predictions run without errors.