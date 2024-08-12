import os
import numpy as np
import torch
import pickle
import argparse

from scipy.fftpack import fft, ifft
from fusion.model import FusionModel
from rf.model import RF_conv_decoder
from rgb.model import CNN3D
from data.datasets import RFDataRAMVersion, RGBData
import matplotlib.pyplot as plt
from utils.utils import extract_video, pulse_rate_from_power_spectral_density

# def parseArgs():
#     parser = argparse.ArgumentParser(description='Configs for running demo fusion script')

#     parser.add_argument('--folds-path', type=str,
#                         default="/Users/jamesemilian/triage/equipleth/Camera_77GHzRadar_Plethysmography_2/demo_fold.pkl",
#                         help='Pickle file containing the folds.')
                        
#     parser.add_argument('--fold', type=int, default=0,
#                         help='Fold Number')
    
#     parser.add_argument('--rgb-dir', type=str,
#                         default="/Users/jamesemilian/triage/equipleth/Camera_77GHzRadar_Plethysmography_2/rgb_files/v_1_1",
#                         help="Directory containing the RGB files")
    
#     parser.add_argument('--rf-dir', type=str,
#                         default="/Users/jamesemilian/triage/equipleth/Camera_77GHzRadar_Plethysmography_2/rf_files/1_1",
#                         help="Directory containing the RF files")
    
#     parser.add_argument('--device', type=str, default=None,
#                         help="Device to run the model on. If not specified, defaults to available device.")

#     parser.add_argument('--rgb-checkpoint-path', type=str,
#                         default="/Users/jamesemilian/triage/equipleth/Camera_77GHzRadar_Plethysmography_2/best_pth/RGB_CNN3D/best.pth",
#                         help="Path to the RGB model checkpoint.")
    
#     parser.add_argument('--rf-checkpoint-path', type=str,
#                         default="/Users/jamesemilian/triage/equipleth/Camera_77GHzRadar_Plethysmography_2/best_pth/RF_IQ_Net/best.pth",
#                         help="Path to the RF model checkpoint.")
    
#     parser.add_argument('--fusion-checkpoint-path', type=str,
#                         default="/Users/jamesemilian/triage/equipleth/Camera_77GHzRadar_Plethysmography_2/best_pth/Fusion/Gen_9_epochs.pth",
#                         help="Path to the fusion model checkpoint.")

#     return parser.parse_args()


def load_rgb_data(folds_path, rgb_dir):
    """Load RGB frames and corresponding ground truth PPG using the RGBData class."""
    
    with open(folds_path, "rb") as fp:
        files_in_fold = pickle.load(fp)

    demo_files = files_in_fold[0]["test"][:1]

    rgb_dataset = RGBData(datapath=rgb_dir, datapaths=demo_files)
    print(f"Loading RGB data from {rgb_dir}")
    # print(f"Files found: {os.listdir(rgb_dir)}")
    
    # if len(rgb_dataset.video_list) == 0:
    #     raise ValueError(f"No valid RGB data found in directory: {rgb_dir}")
    return rgb_dataset

def load_rf_data(folds_path, rf_dir):
    """Load RF data and corresponding ground truth PPG using the RFDataRAMVersion class."""

    with open(folds_path, "rb") as fp:
        files_in_fold = pickle.load(fp)

    demo_files = files_in_fold[0]["test"][:1]
    demo_files = [i[2:] for i in demo_files]
    print(demo_files)

    rf_dataset = RFDataRAMVersion(datapath=rf_dir, datapaths=demo_files, frame_length_ppg = 128, 
                                    static_dataset_samples=15)

    print(f"Loading RF data from {rf_dir}")
    # print(f"Files found: {os.listdir(rf_dir)}")
    
    # if len(rf_dataset.rf_file_list) == 0:
    #     raise ValueError(f"No valid RF data found in directory: {rf_dir}")
    return rf_dataset

def preprocess_rgb(frames, model, device, sequence_length=64):
    """Process RGB frames to estimate rPPG using the RGB model."""
    model.eval()
    estimated_ppg = None
    cur_cat_frames = None
    
    #Check if this logic works.
    #also verify if the loaded rgb dataset is a set of frames as expected.
    #For this to work on multiple videos; need to include the 'for cur_session in session_names' as seen in eval.py

    for cur_frame_num in range(len(frames)):
        cur_frame = frames[cur_frame_num]
        print(f"Current frame has shape {cur_frame.shape}")
        print(f"Length of frames is: {len(frames)}")
        cur_frame = torch.from_numpy(cur_frame.astype(np.uint8)).permute(2, 0, 1).float() / 255.0 #reshapes hxwxc to cxhxw
        cur_frame = cur_frame.unsqueeze(0).to(device) #becomes 1xcxhxw
        
        if cur_frame_num % sequence_length == 0:
            cur_cat_frames = cur_frame
        else:
            cur_cat_frames = torch.cat((cur_cat_frames, cur_frame), 0)
        
        if cur_cat_frames.shape[0] == sequence_length:
            cur_cat_frames = cur_cat_frames.unsqueeze(0)
            cur_cat_frames = torch.transpose(cur_cat_frames, 1, 2)
            
            with torch.no_grad():
                cur_est_ppg, _, _, _ = model(cur_cat_frames)
                cur_est_ppg = cur_est_ppg.squeeze().cpu().numpy()
                
                if estimated_ppg is None:
                    estimated_ppg = cur_est_ppg
                else:
                    estimated_ppg = np.concatenate((estimated_ppg, cur_est_ppg), -1)
                    
    return estimated_ppg
    #remember to write in eval code from the latter half of eval.eval_rgb_model.
    #for now this is purely an eval script.

def preprocess_rf(rf_signal, model, device, sequence_length=128, sampling_ratio=4):
    """Process RF signal to estimate rPPG using the RF model."""
    model.eval()
    estimated_ppg = None
    cur_cat_frames = None

    for cur_frame_num in range(rf_signal.shape[-1]):
        cur_frame = rf_signal[:, :, cur_frame_num]
        cur_frame = torch.tensor(cur_frame).float().to(device)
        
        if cur_frame_num % (sequence_length * sampling_ratio) == 0:
            cur_cat_frames = cur_frame.unsqueeze(0)
        else:
            cur_cat_frames = torch.cat((cur_cat_frames, cur_frame.unsqueeze(0)), 0)
        
        if cur_cat_frames.shape[0] == sequence_length * sampling_ratio:
            cur_cat_frames = cur_cat_frames.unsqueeze(0)
            cur_cat_frames = torch.transpose(cur_cat_frames, 1, 2)
            cur_cat_frames = torch.transpose(cur_cat_frames, 2, 3)
            IQ_frames = torch.reshape(cur_cat_frames, (cur_cat_frames.shape[0], -1, cur_cat_frames.shape[3]))
            
            with torch.no_grad():
                cur_est_ppg, _ = model(IQ_frames)
                cur_est_ppg = cur_est_ppg.squeeze().cpu().numpy()
                
                if estimated_ppg is None:
                    estimated_ppg = cur_est_ppg
                else:
                    estimated_ppg = np.concatenate((estimated_ppg, cur_est_ppg), -1)
                    
    return estimated_ppg

def calculate_hr(ppg_signal, fs=30):
    """Calculate Heart Rate (HR) from the PPG signal using its FFT."""
    return pulse_rate_from_power_spectral_density(ppg_signal, FS=fs, LL_PR=45, UL_PR=180, BUTTER_ORDER=6)

def calculate_rr(ppg_signal, fs=30):
    """Calculate Respiratory Rate (RR) from the PPG signal using its FFT."""
    return pulse_rate_from_power_spectral_density(ppg_signal, FS=fs, LL_PR=5, UL_PR=50, BUTTER_ORDER=6)

def apply_fft(ppg_signal, fft_resolution=1):
    """Apply FFT to a PPG signal."""
    n_curr = len(ppg_signal) * fft_resolution
    fft_signal = np.abs(fft(ppg_signal, n=int(n_curr), axis=0))
    return fft_signal

def run_inference(folds_path, rgb_dir, rf_dir, rgb_model, rf_model, fusion_model, device):
    """Run inference on RGB and RF data to predict rPPG, HR, and RR."""
    # Load datasets using the classes
    rf_dataset = load_rf_data(folds_path, rf_dir)
    rgb_dataset = load_rgb_data(folds_path, rgb_dir)
    
    print(f"RGB Dataset: {rgb_dataset}")
    print(f"RF Dataset: {rf_dataset}")

    # Preprocess and predict RGB and RF PPG signals
    estimated_rgb_ppg = preprocess_rgb([data[0] for data in rgb_dataset], rgb_model, device)
    estimated_rf_ppg = preprocess_rf([data[0] for data in rf_dataset], rf_model, device)
    
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
    normalized_rgb_gt_ppg = (rgb_dataset.signal_list.flatten() - np.mean(rgb_dataset.signal_list)) / np.std(rgb_dataset.signal_list)
    normalized_rf_gt_ppg = (rf_dataset.signal_list.flatten() - np.mean(rf_dataset.signal_list)) / np.std(rf_dataset.signal_list)
    
    return predicted_rppg, hr, rr, normalized_rgb_gt_ppg, normalized_rf_gt_ppg
    # return normalized_rf_gt_ppg #debug version

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
    rgb_dir = "/Users/jamesemilian/triage/equipleth/Camera_77GHzRadar_Plethysmography_2/rgb_files"  # Example path to RGB data
    rf_dir = "/Users/jamesemilian/triage/equipleth/Camera_77GHzRadar_Plethysmography_2/rf_files"    # Example path to RF data
    rgb_checkpoint_path = "/Users/jamesemilian/triage/equipleth/Camera_77GHzRadar_Plethysmography_2/best_pth/RGB_CNN3D/best.pth"  # Example checkpoint for RGB model
    rf_checkpoint_path = "/Users/jamesemilian/triage/equipleth/Camera_77GHzRadar_Plethysmography_2/best_pth/RF_IQ_Net/best.pth"   # Example checkpoint for RF model
    fusion_checkpoint_path = "/Users/jamesemilian/triage/equipleth/Camera_77GHzRadar_Plethysmography_2/best_pth/Fusion/Gen_9_epochs.pth"  # Example checkpoint for Fusion model
    folds_path = "/Users/jamesemilian/triage/equipleth/Camera_77GHzRadar_Plethysmography_2/demo_fold.pkl"

    # destination_folder = "/Users/jamesemilian/triage/equipleth/Camera_77GHzRadar_Plethysmography_2/rgb_files"

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
    predicted_rppg, hr, rr, rgb_gt_ppg, rf_gt_ppg = run_inference(folds_path, rgb_dir, rf_dir, rgb_model, rf_model, fusion_model, device)
    # rf_gt_ppg = run_inference(rgb_dir, rf_dir, rgb_model, rf_model, fusion_model, device) #debug version
    
    # Output results
    print(f"Predicted HR: {hr} bpm")
    print(f"Predicted RR: {rr} bpm")
    # Plotting the results
    plt.figure(figsize=(12, 6))

    # Plot predicted PPG
    plt.subplot(3, 1, 1)
    plt.plot(predicted_rppg, label='Predicted PPG')
    plt.title('Predicted PPG Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()

    # Plot ground truth RGB PPG
    plt.subplot(3, 1, 2)
    plt.plot(rgb_gt_ppg, label='Ground Truth RGB PPG', color='g')
    plt.title('Ground Truth RGB PPG Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()

    # Plot ground truth RF PPG
    plt.subplot(3, 1, 3)
    plt.plot(rf_gt_ppg, label='Ground Truth RF PPG', color='r')
    plt.title('Ground Truth RF PPG Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
