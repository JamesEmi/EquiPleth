import os
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from torch.utils.data import Dataset
import rf.organizer as org
from rf.proc import create_fast_slow_matrix, find_range
from rgb.model import CNN3D
from rf.model import RF_conv_decoder
from fusion.model import FusionModel
from utils.utils import extract_video, pulse_rate_from_power_spectral_density

# Step 1: Generate a small pickle file from a single RGB and RF data sample

def gen_single_sample_pickle(rgb_dir, rf_dir, rgb_model, rf_model, save_path, device):
    """Generate a small pickle file from a single RGB and RF data sample."""
    rgb_model.eval()
    rf_model.eval()

    # Process the RGB data
    rgb_sample_name = os.listdir(rgb_dir)[0] #should use the fold and pull from there but this works too, for now
    cur_video_path = os.path.join(rgb_dir, rgb_sample_name)
    print(f'Processing RGB sample from {cur_video_path}')
    
    frames = extract_video(path=cur_video_path, file_str="rgbd_rgb")
    circ_buff = frames[0:100]
    frames = np.concatenate((frames, circ_buff))
    estimated_rgb_ppgs = None

    for cur_frame_num in range(frames.shape[0]):
        cur_frame = frames[cur_frame_num, :, :, :]
        cur_frame_cropped = torch.from_numpy(cur_frame.astype(np.uint8)).permute(2, 0, 1).float()
        cur_frame_cropped = cur_frame_cropped / 255
        cur_frame_cropped = cur_frame_cropped.unsqueeze(0).to(device)

        if cur_frame_num % 64 == 0:
            cur_cat_frames = cur_frame_cropped
        else:
            cur_cat_frames = torch.cat((cur_cat_frames, cur_frame_cropped), 0)

        if cur_cat_frames.shape[0] == 64:
            cur_cat_frames = cur_cat_frames.unsqueeze(0)
            cur_cat_frames = torch.transpose(cur_cat_frames, 1, 2)
            with torch.no_grad():
                cur_est_ppg, _, _, _ = rgb_model(cur_cat_frames)
                cur_est_ppg = cur_est_ppg.squeeze().cpu().numpy()

            if estimated_rgb_ppgs is None:
                estimated_rgb_ppgs = cur_est_ppg
            else:
                estimated_rgb_ppgs = np.concatenate((estimated_rgb_ppgs, cur_est_ppg), -1)
    
    rgb_gt_ppg = np.load(os.path.join(cur_video_path, "rgbd_ppg.npy"))[:900]

    # Process the RF data
    rf_sample_name = os.listdir(rf_dir)[0]
    cur_rf_path = os.path.join(rf_dir, rf_sample_name)
    print(f'Processing RF sample from {cur_rf_path}')

    rf_fptr = open(os.path.join(cur_rf_path, "rf.pkl"), 'rb')
    s = pickle.load(rf_fptr)
    adc_samples = 256
    freq_slope = 60.012e12
    samp_f = 5e6
    rf_window_size = 5

    rf_organizer = org.Organizer(s, 1, 1, 1, 2 * adc_samples)
    frames = rf_organizer.organize()
    frames = frames[:,:,:,0::2]  # Remove alternate zeros

    data_f = create_fast_slow_matrix(frames)
    range_index = find_range(data_f, samp_f, freq_slope, adc_samples)
    temp_window = np.blackman(rf_window_size)
    raw_data = data_f[:, range_index - len(temp_window) // 2 : range_index + len(temp_window) // 2 + 1]
    circ_buffer = raw_data[0:800]
    raw_data = np.concatenate((raw_data, circ_buffer))
    raw_data = np.array([np.real(raw_data), np.imag(raw_data)])
    raw_data = np.transpose(raw_data, axes=(0, 2, 1))
    rf_data = raw_data
    rf_data = np.transpose(rf_data, axes=(2, 0, 1))

    estimated_rf_ppgs = None
    for cur_frame_num in range(rf_data.shape[0]):
        cur_frame = rf_data[cur_frame_num, :, :]
        cur_frame = torch.tensor(cur_frame).float().to(device) / 1.255e5
        
        if cur_frame_num % (128 * 4) == 0:
            cur_cat_frames = cur_frame.unsqueeze(0)
        else:
            cur_cat_frames = torch.cat((cur_cat_frames, cur_frame.unsqueeze(0)), 0)
        
        if cur_cat_frames.shape[0] == 128 * 4:
            cur_cat_frames = cur_cat_frames.unsqueeze(0)
            cur_cat_frames = torch.transpose(cur_cat_frames, 1, 2)
            cur_cat_frames = torch.transpose(cur_cat_frames, 2, 3)
            IQ_frames = torch.reshape(cur_cat_frames, (cur_cat_frames.shape[0], -1, cur_cat_frames.shape[3]))
            
            with torch.no_grad():
                cur_est_ppg, _ = rf_model(IQ_frames)
                cur_est_ppg = cur_est_ppg.squeeze().cpu().numpy()
                
                if estimated_rf_ppgs is None:
                    estimated_rf_ppgs = cur_est_ppg
                else:
                    estimated_rf_ppgs = np.concatenate((estimated_rf_ppgs, cur_est_ppg), -1)
    
    rf_gt_ppg = np.load(os.path.join(cur_rf_path, "vital_dict.npy"), allow_pickle=True).item()['rgbd']['NOM_PLETHWaveExport'][:900]

    # Save the results into a small pickle file
    sample_data = [{
        'video_path': rgb_sample_name,
        'est_ppgs': estimated_rgb_ppgs[:900],
        'gt_ppgs': rgb_gt_ppg,
        'rf_ppg': estimated_rf_ppgs[:900]
    }]
    
    with open(save_path, 'wb') as handle:
        pickle.dump(sample_data, handle)
    
    print(f"Sample data saved at {save_path}")

# Step 2: Create a simplified dataset class to load the single sample for inference

class SingleSampleFusionDataset(Dataset):
    def __init__(self, pickle_path, compute_fft=True, fs=30, l_freq_bpm=45, u_freq_bpm=180, fft_resolution=1):
        self.compute_fft = compute_fft
        self.fs = fs
        self.l_freq_bpm = l_freq_bpm
        self.u_freq_bpm = u_freq_bpm
        self.fft_resolution = fft_resolution
        
        # Load the data from the small pickle file
        with open(pickle_path, 'rb') as f:
            self.sample_data = pickle.load(f)[0]

        self.ppg_offset = 25
        self.rgb_ppg = self.sample_data['est_ppgs'][self.ppg_offset:]
        self.rf_ppg = self.sample_data['rf_ppg'][self.ppg_offset:]
        self.gt_ppg = self.sample_data['gt_ppgs'][self.ppg_offset:]
        
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        item = {'est_ppgs': self.rgb_ppg, 'rf_ppg': self.rf_ppg}
        item_sig = self.gt_ppg

        item_sig = (item_sig - np.mean(item_sig)) / np.std(item_sig)
        item['est_ppgs'] = (item['est_ppgs'] - np.mean(item['est_ppgs'])) / np.std(item['est_ppgs'])
        item['rf_ppg'] = (item['rf_ppg'] - np.mean(item['rf_ppg'])) / np.std(item['rf_ppg'])

        if self.compute_fft:
            n_curr = len(item_sig) * self.fft_resolution
            fft_gt = np.abs(np.fft.fft(item_sig, n=int(n_curr), axis=0))
            fft_gt = fft_gt / np.max(fft_gt, axis=0)
            
            fft_est = np.abs(np.fft.fft(item['est_ppgs'], n=int(n_curr), axis=0))
            fft_est = fft_est / np.max(fft_est, axis=0)
            fft_est = fft_est[self.l_freq_idx : self.u_freq_idx + 1]

            fft_rf = np.abs(np.fft.fft(item['rf_ppg'], n=int(n_curr), axis=0))
            fft_rf = fft_rf / np.max(fft_rf, axis=0)
            fft_rf = fft_rf[self.l_freq_idx : self.u_freq_idx + 1]

            return {'est_ppgs': fft_est, 'rf_ppg': fft_rf}, fft_gt[self.l_freq_idx : self.u_freq_idx + 1]
        else:
            return item, np.array(item_sig)

# Step 3: Modify the main demo_fusion script to use the above methods

def run_inference(folds_path, rgb_dir, rf_dir, rgb_model, rf_model, fusion_model, device):
    """Run inference on RGB and RF data to predict rPPG, HR, and RR."""
    # Generate the small pickle file
    pickle_path = "sample_data.pkl"
    gen_single_sample_pickle(rgb_dir, rf_dir, rgb_model, rf_model, pickle_path, device)

    # Load the dataset from the small pickle file
    dataset = SingleSampleFusionDataset(pickle_path=pickle_path)

    # Preprocess and predict RGB and RF PPG signals
    sample_data, fft_gt = dataset[0]
    
    # Convert to torch tensors
    rgb_fft_tensor = torch.tensor(sample_data['est_ppgs'], dtype=torch.float32).unsqueeze(0).to(device)
    print(f"Shape of rgb_fft_tensor is: {rgb_fft_tensor.shape}")
    rf_fft_tensor = torch.tensor(sample_data['rf_ppg'], dtype=torch.float32).unsqueeze(0).to(device)
    print(f"Shape of rf_fft_tensor is: {rf_fft_tensor.shape}")
    
    # Model inference
    fusion_model.eval()
    with torch.no_grad():
        predicted_fft = fusion_model(rgb_fft_tensor, rf_fft_tensor)
    
    # Post-process: Reconstruct rPPG signal using IFFT
    predicted_fft = predicted_fft.squeeze().cpu().numpy()
    predicted_rppg = np.real(ifft(predicted_fft))
    
    # Calculate HR and RR from the predicted rPPG signal
    hr = pulse_rate_from_power_spectral_density(predicted_rppg, FS=30, LL_PR=45, UL_PR=180, BUTTER_ORDER=6)
    rr = pulse_rate_from_power_spectral_density(predicted_rppg, FS=30, LL_PR=5, UL_PR=50, BUTTER_ORDER=6)
    
    return predicted_rppg, hr, rr, fft_gt

def main():
    # Paths
    rgb_dir = "/Users/jamesemilian/triage/equipleth/Camera_77GHzRadar_Plethysmography_2/rgb_files"  # Example path to RGB data
    rf_dir = "/Users/jamesemilian/triage/equipleth/Camera_77GHzRadar_Plethysmography_2/rf_files"    # Example path to RF data
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
    predicted_rppg, hr, rr, fft_gt = run_inference(None, rgb_dir, rf_dir, rgb_model, rf_model, fusion_model, device)
    
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

    # Plot ground truth FFT
    plt.subplot(3, 1, 2)
    plt.plot(fft_gt, label='Ground Truth FFT', color='g')
    plt.title('Ground Truth FFT')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
