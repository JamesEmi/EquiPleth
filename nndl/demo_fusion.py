import os
import numpy as np
import torch
import pickle
from scipy.fftpack import fft, ifft
from fusion.model import FusionModel
from rf.proc import create_fast_slow_matrix, find_range
import matplotlib.pyplot as plt

def load_rgb_data(rgb_dir):
    """Load RGB frames and corresponding ground truth PPG."""
    frames = []
    for img_file in sorted(os.listdir(rgb_dir)):
        if img_file.endswith('.png'):
            img_path = os.path.join(rgb_dir, img_file)
            frames.append(plt.imread(img_path))
    
    frames = np.array(frames)
    # Load ground truth PPG
    gt_ppg = np.load(os.path.join(rgb_dir, 'rgbd_ppg.npy'))
    return frames, gt_ppg

def load_rf_data(rf_dir):
    """Load RF data and corresponding ground truth PPG."""
    with open(os.path.join(rf_dir, "rf.pkl"), 'rb') as rf_file:
        rf_data = pickle.load(rf_file)
    
    # Organize the RF data
    rf_organizer = create_fast_slow_matrix(rf_data)
    adc_samples = 256
    samp_f = 5e6
    freq_slope = 60.012e12
    range_index = find_range(rf_organizer, samp_f, freq_slope, adc_samples)
    window_size = 5
    rf_window = np.blackman(window_size)
    rf_signal = rf_organizer[:, range_index-len(rf_window)//2:range_index+len(rf_window)//2 + 1]
    rf_signal = np.array([np.real(rf_signal), np.imag(rf_signal)])
    rf_signal = np.transpose(rf_signal, axes=(0, 2, 1))
    
    # Load ground truth PPG
    gt_ppg = np.load(os.path.join(rf_dir, 'vital_dict.npy'), allow_pickle=True).item()['rgbd']['NOM_PLETHWaveExport']
    return rf_signal, gt_ppg

def preprocess_rgb(frames, model, device, sequence_length=64):
    """Process RGB frames to estimate rPPG using the RGB model."""
    estimated_ppg = []
    for i in range(0, len(frames) - sequence_length, sequence_length):
        batch = frames[i:i + sequence_length]
        batch = torch.tensor(batch).float().permute(0, 3, 1, 2) / 255.0  # Normalize and change dims
        batch = batch.unsqueeze(0).to(device)  # Add batch dimension
        with torch.no_grad():
            ppg_est, _, _, _ = model(batch)
            estimated_ppg.append(ppg_est.squeeze().cpu().numpy())
    estimated_ppg = np.concatenate(estimated_ppg, axis=0)
    return estimated_ppg

def preprocess_rf(rf_signal, model, device, sequence_length=128, sampling_ratio=4):
    """Process RF signal to estimate rPPG using the RF model."""
    estimated_ppg = []
    for i in range(0, rf_signal.shape[-1] - sequence_length * sampling_ratio, sequence_length * sampling_ratio):
        batch = rf_signal[:, :, i:i + sequence_length * sampling_ratio]
        batch = torch.tensor(batch).float().to(device)
        with torch.no_grad():
            ppg_est, _ = model(batch)
            estimated_ppg.append(ppg_est.squeeze().cpu().numpy())
    estimated_ppg = np.concatenate(estimated_ppg, axis=0)
    return estimated_ppg

def apply_fft(ppg_signal, fft_resolution=1):
    """Apply FFT to a PPG signal."""
    n_curr = len(ppg_signal) * fft_resolution
    fft_signal = np.abs(fft(ppg_signal, n=int(n_curr), axis=0))
    return fft_signal

def run_inference(rgb_dir, rf_dir, rgb_model, rf_model, fusion_model, device):
    """Run inference on RGB and RF data to predict rPPG, HR, and RR."""
    # Load and preprocess data
    rgb_frames, rgb_gt_ppg = load_rgb_data(rgb_dir)
    rf_signal, rf_gt_ppg = load_rf_data(rf_dir)
    
    estimated_rgb_ppg = preprocess_rgb(rgb_frames, rgb_model, device)
    estimated_rf_ppg = preprocess_rf(rf_signal, rf_model, device)
    
    # Apply FFT
    rgb_fft = apply_fft(estimated_rgb_ppg)
    rf_fft = apply_fft(estimated_rf_ppg)
    
    # Convert to torch tensors
    rgb_fft_tensor = torch.tensor(rgb_fft, dtype=torch.float32).unsqueeze(0).to(device)
    rf_fft_tensor = torch.tensor(rf_fft, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Model inference
    fusion_model.eval()
    with torch.no_grad():
        predicted_fft = fusion_model(rgb_fft_tensor, rf_fft_tensor)
    
    # Post-process: Reconstruct rPPG signal using IFFT
    predicted_fft = predicted_fft.squeeze().cpu().numpy()
    predicted_rppg = np.real(ifft(predicted_fft))
    
    # Calculate HR and RR from the predicted rPPG signal
    hr = calculate_hr(predicted_rppg)
    rr = calculate_rr(predicted_rppg)
    
    # For comparison, normalize the ground truth PPG signal
    normalized_rgb_gt_ppg = (rgb_gt_ppg - np.mean(rgb_gt_ppg)) / np.std(rgb_gt_ppg)
    normalized_rf_gt_ppg = (rf_gt_ppg - np.mean(rf_gt_ppg)) / np.std(rf_gt_ppg)
    
    return predicted_rppg, hr, rr, normalized_rgb_gt_ppg, normalized_rf_gt_ppg

def calculate_hr(ppg_signal, fs=30):
    """Calculate Heart Rate (HR) from the PPG signal using its FFT."""
    freqs = np.fft.fftfreq(len(ppg_signal), d=1/fs) * 60
    ppg_fft = np.abs(np.fft.fft(ppg_signal))
    peak_freq = freqs[np.argmax(ppg_fft)]
    return peak_freq

def calculate_rr(ppg_signal, fs=30):
    """Calculate Respiratory Rate (RR) from the PPG signal using its FFT."""
    freqs = np.fft.fftfreq(len(ppg_signal), d=1/fs) * 60
    ppg_fft = np.abs(np.fft.fft(ppg_signal))
    peak_freq = freqs[np.argmax(ppg_fft)]
    return peak_freq / 4  # Simplified ratio for RR (adjust based on actual data characteristics)

def main():
    # Arguments
    rgb_dir = "/Users/jamesemilian/triage/equipleth/Camera_77GHzRadar_Plethysmography_2/rgb_files/v_1_1"  # Example path to RGB data
    rf_dir = "/Users/jamesemilian/triage/equipleth/Camera_77GHzRadar_Plethysmography_2/rf_files/1_1"    # Example path to RF data
    rgb_checkpoint_path = "/Users/jamesemilian/triage/equipleth/Camera_77GHzRadar_Plethysmography_2/best_pth/RGB_CNN3D/best.pth"  # Example checkpoint for RGB model
    rf_checkpoint_path = "/Users/jamesemilian/triage/equipleth/Camera_77GHzRadar_Plethysmography_2/best_pth/RF_IQ_Net/best.pth"   # Example checkpoint for RF model
    fusion_checkpoint_path = "/Users/jamesemilian/triage/equipleth/Camera_77GHzRadar_Plethysmography_2/best_pth/Fusion/Gen_9_epochs.pth"  # Example checkpoint for Fusion model
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load models
    rgb_model = CNN3D().to(device)
    rgb_model.load_state_dict(torch.load(rgb_checkpoint_path))
    
    rf_model = RF_conv_decoder().to(device)
    rf_model.load_state_dict(torch.load(rf_checkpoint_path))
    
    fusion_model = FusionModel(base_ppg_est_len=1024, rf_ppg_est_len=1024*5, out_len=1024)
    fusion_model.load_state_dict(torch.load(fusion_checkpoint_path))
    fusion_model.to(device)
    
    # Run inference
    predicted_rppg, hr, rr, rgb_gt_ppg, rf_gt_ppg = run_inference(rgb_dir, rf_dir, rgb_model, rf_model, fusion_model, device)
    
    # Output results
    print(f"Predicted HR: {hr} bpm")
    print(f"Predicted RR: {rr} bpm")
    print(f"Ground Truth PPG (RGB): {rgb_gt_ppg}")
    print(f"Ground Truth PPG (RF): {rf_gt_ppg}")

if __name__ == '__main__':
    main()
