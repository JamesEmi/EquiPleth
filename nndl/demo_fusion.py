import os
import numpy as np
import torch
import pickle
from scipy.fftpack import fft, ifft
from fusion.model import FusionModel
from rf.model import RF_conv_decoder
from rgb.model import CNN3D
from data.datasets import RFDataRAMVersion, RGBData
import matplotlib.pyplot as plt

def load_rgb_data(rgb_dir):
    """Load RGB frames and corresponding ground truth PPG using the RGBData class."""
    rgb_dataset = RGBData(datapath=rgb_dir, datapaths=os.listdir(rgb_dir))
    print(f"Loading RGB data from {rgb_dir}")
    print(f"Files found: {os.listdir(rgb_dir)}")
    
    if len(rgb_dataset.video_list) == 0:
        raise ValueError(f"No valid RGB data found in directory: {rgb_dir}")
    return rgb_dataset

def load_rf_data(rf_dir):
    """Load RF data and corresponding ground truth PPG using the RFDataRAMVersion class."""
    rf_dataset = RFDataRAMVersion(datapath=rf_dir, datapaths=os.listdir(rf_dir))
    print(f"Loading RF data from {rf_dir}")
    print(f"Files found: {os.listdir(rf_dir)}")
    
    if len(rf_dataset.rf_file_list) == 0:
        raise ValueError(f"No valid RF data found in directory: {rf_dir}")
    return rf_dataset

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
    # Load datasets using the classes
    rf_dataset = load_rf_data(rf_dir)
    # rgb_dataset = load_rgb_data(rgb_dir)
    
    # Preprocess and predict RGB and RF PPG signals
    # estimated_rgb_ppg = preprocess_rgb([data[0] for data in rgb_dataset], rgb_model, device)
    estimated_rf_ppg = preprocess_rf([data[0] for data in rf_dataset], rf_model, device)
    
    # Apply FFT
    # rgb_fft = apply_fft(estimated_rgb_ppg)
    rf_fft = apply_fft(estimated_rf_ppg)
    
    # Convert to torch tensors
    # rgb_fft_tensor = torch.tensor(rgb_fft, dtype=torch.float32).unsqueeze(0).to(device)
    rf_fft_tensor = torch.tensor(rf_fft, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Model inference
    # fusion_model.eval()
    # with torch.no_grad():
    #     predicted_fft = fusion_model(rgb_fft_tensor, rf_fft_tensor)
    
    # Post-process: Reconstruct rPPG signal using IFFT
    # predicted_fft = predicted_fft.squeeze().cpu().numpy()
    # predicted_rppg = np.real(ifft(predicted_fft))
    
    # Calculate HR and RR from the predicted rPPG signal
    # hr = calculate_hr(predicted_rppg)
    # rr = calculate_rr(predicted_rppg)
    
    # For comparison, normalize the ground truth PPG signal
    # normalized_rgb_gt_ppg = (rgb_dataset.signal_list.flatten() - np.mean(rgb_dataset.signal_list)) / np.std(rgb_dataset.signal_list)
    normalized_rf_gt_ppg = (rf_dataset.signal_list.flatten() - np.mean(rf_dataset.signal_list)) / np.std(rf_dataset.signal_list)
    
    # return predicted_rppg, hr, rr, normalized_rgb_gt_ppg, normalized_rf_gt_ppg
    return normalized_rf_gt_ppg #debug version

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
    rgb_model.load_state_dict(torch.load(rgb_checkpoint_path, map_location=device))
    
    rf_model = RF_conv_decoder().to(device)
    rf_model.load_state_dict(torch.load(rf_checkpoint_path, map_location=device))
    
    fusion_model = FusionModel(base_ppg_est_len=1024, rf_ppg_est_len=1024*5, out_len=1024)
    fusion_model.load_state_dict(torch.load(fusion_checkpoint_path, map_location=device))
    fusion_model.to(device)
    
    # Run inference
    # predicted_rppg, hr, rr, rgb_gt_ppg, rf_gt_ppg = run_inference(rgb_dir, rf_dir, rgb_model, rf_model, fusion_model, device)
    rf_gt_ppg = run_inference(rgb_dir, rf_dir, rgb_model, rf_model, fusion_model, device) #debug version
    
    # Output results
    # print(f"Predicted HR: {hr} bpm")
    # print(f"Predicted RR: {rr} bpm")
    # print(f"Ground Truth PPG (RGB): {rgb_gt_ppg}")
    print(f"Ground Truth PPG (RF): {rf_gt_ppg}")

if __name__ == '__main__':
    main()
