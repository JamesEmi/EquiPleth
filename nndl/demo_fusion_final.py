import os
import pickle
import numpy as np
import torch
import argparse
import imageio
from tqdm import tqdm

from fusion.model import FusionModel
from rgb.model import CNN3D
from rf.model import RF_conv_decoder
from utils.eval_fusion_je import eval_fusion_model
from utils.utils_je import pulse_rate_from_power_spectral_density, extract_video
from data.datasets_je import FusionEvalDatasetObject, FusionRunDatasetObject
from rf import organizer as org
from rf.proc import create_fast_slow_matrix, find_range
import matplotlib.pyplot as plt
import datetime

def parseArgs():
    parser = argparse.ArgumentParser(description='Script to generate PPGs and test one sample using the fusion model')
    parser.add_argument('-rgb-dir', '--rgb-dir', default="/Users/jamesemilian/triage/equipleth/Camera_77GHzRadar_Plethysmography_2/rgb_files",
                        type=str, help="Directory containing RGB frames.")
    parser.add_argument('-rf-file', '--rf-dir', default="/Users/jamesemilian/triage/equipleth/Camera_77GHzRadar_Plethysmography_2/rf_files", 
                        type=str, help="Path to rf.pkl file.") #/Users/jamesemilian/triage/equipleth/Camera_77GHzRadar_Plethysmography_2/rf_files
    parser.add_argument('--device', type=str, default=None, help="Device to run the model on.")
    parser.add_argument('--sample-idx', type=int, default=0, help="Index of the sample to test.")
    parser.add_argument('--verbose', action='store_true', help="Verbosity.")

    return parser.parse_args()

def extract_rgb_frames(rgb_dir):
    """Extract RGB frames from the provided directory."""
    frames = []
    for file_name in sorted(os.listdir(rgb_dir)):
        if file_name.endswith('.png'):
            file_path = os.path.join(rgb_dir, file_name)
            frame = imageio.imread(file_path)
            frames.append(frame)
    return np.array(frames)

def gen_rgb_preds(root_dir, session_name, model, sequence_length=64, max_ppg_length = 900, file_name = "rgbd_rgb", device=torch.device('cpu')):
    model.eval()
    video_samples = []

    for cur_session in session_name:
        single_video_sample = {"video_path" : os.path.join(root_dir, cur_session)}
        video_samples.append(single_video_sample)

    for cur_video_sample in tqdm(video_samples):
        cur_video_path = cur_video_sample["video_path"]
        cur_est_ppgs = None
        frames = extract_video(path=cur_video_path, file_str=file_name)
        circ_buff = frames[0:100]
        frames = np.concatenate((frames, circ_buff))


        for cur_frame_num in range(frames.shape[0]):
                    # Preprocess
                    cur_frame = frames[cur_frame_num, :, :, :]
                    cur_frame_cropped = torch.from_numpy(cur_frame.astype(np.uint8)).permute(2, 0, 1).float()
                    cur_frame_cropped = cur_frame_cropped / 255
                    # Add the T dim
                    cur_frame_cropped = cur_frame_cropped.unsqueeze(0).to(device) 

                    # Concat
                    if cur_frame_num % sequence_length == 0:
                        cur_cat_frames = cur_frame_cropped
                    else:
                        cur_cat_frames = torch.cat((cur_cat_frames, cur_frame_cropped), 0)

                    # Test the performance
                    if cur_cat_frames.shape[0] == sequence_length:
                        
                        # DL
                        with torch.no_grad():
                            # Add the B dim
                            cur_cat_frames = cur_cat_frames.unsqueeze(0) 
                            cur_cat_frames = torch.transpose(cur_cat_frames, 1, 2)
                            # Get the estimated PPG signal
                            cur_est_ppg, _, _, _ = model(cur_cat_frames)
                            cur_est_ppg = cur_est_ppg.squeeze().cpu().numpy()

                        # First sequence
                        if cur_est_ppgs is None: 
                            cur_est_ppgs = cur_est_ppg
                        else:
                            cur_est_ppgs = np.concatenate((cur_est_ppgs, cur_est_ppg), -1)

        cur_video_sample['video_path'] = os.path.basename(cur_video_sample['video_path'])                    
        cur_video_sample['est_ppg'] = cur_est_ppgs[0:max_ppg_length]

    return video_samples

def gen_rf_preds(root_path, demo_files, model, sequence_length = 128, max_ppg_length = 900, 
                  adc_samples = 256, rf_window_size = 5, freq_slope=60.012e12, 
                  samp_f=5e6, sampling_ratio = 4, device=torch.device('cpu')):
    model.eval()
    video_samples = []
    for rf_folder in tqdm(demo_files, total=len(demo_files)):
        try:
            rf_fptr = open(os.path.join(root_path, rf_folder, "rf.pkl"),'rb')
            s = pickle.load(rf_fptr)
            # Number of samples is set ot 256 for our experiments
            rf_organizer = org.Organizer(s, 1, 1, 1, 2*adc_samples) 
            frames = rf_organizer.organize()
            # The RF read adds zero alternatively to the samples. Remove these zeros.
            frames = frames[:,:,:,0::2] 

            data_f = create_fast_slow_matrix(frames)
            range_index = find_range(data_f, samp_f, freq_slope, adc_samples)
            temp_window = np.blackman(rf_window_size)
            raw_data = data_f[:, range_index-len(temp_window)//2:range_index+len(temp_window)//2 + 1]
            circ_buffer = raw_data[0:800]
            
            # Concatenate extra to generate ppgs of size 3600
            raw_data = np.concatenate((raw_data, circ_buffer))
            raw_data = np.array([np.real(raw_data),  np.imag(raw_data)])
            raw_data = np.transpose(raw_data, axes=(0,2,1))
            rf_data = raw_data

            rf_data = np.transpose(rf_data, axes=(2,0,1))
            cur_video_sample = {}

            cur_est_ppgs = None

            for cur_frame_num in range(rf_data.shape[0]):
                # Preprocess
                cur_frame = rf_data[cur_frame_num, :, :]
                cur_frame = torch.tensor(cur_frame).type(torch.float32)/1.255e5
                # Add the T dim
                cur_frame = cur_frame.unsqueeze(0).to(device)

                # Concat
                if cur_frame_num % (sequence_length*sampling_ratio) == 0:
                    cur_cat_frames = cur_frame
                else:
                    cur_cat_frames = torch.cat((cur_cat_frames, cur_frame), 0)

                # Test the performance
                if cur_cat_frames.shape[0] == sequence_length*sampling_ratio:
                    
                    # DL
                    with torch.no_grad():
                        # Add the B dim
                        cur_cat_frames = cur_cat_frames.unsqueeze(0)
                        cur_cat_frames = torch.transpose(cur_cat_frames, 1, 2)
                        cur_cat_frames = torch.transpose(cur_cat_frames, 2, 3)
                        IQ_frames = torch.reshape(cur_cat_frames, (cur_cat_frames.shape[0], -1, cur_cat_frames.shape[3]))
                        cur_est_ppg, _ = model(IQ_frames)
                        cur_est_ppg = cur_est_ppg.squeeze().cpu().numpy()

                    # First seq
                    if cur_est_ppgs is None: 
                        cur_est_ppgs = cur_est_ppg
                    else:
                        cur_est_ppgs = np.concatenate((cur_est_ppgs, cur_est_ppg), -1)
        
            # Save
            cur_video_sample['video_path'] = rf_folder
            cur_video_sample['rf_ppg'] = cur_est_ppgs[0:max_ppg_length]

            video_samples.append(cur_video_sample)
        except:
            if args.verbose:
                print("RF folder does not exist : ", rf_folder)
    if args.verbose:
        print('All finished!')
    return video_samples


def save_fusion_data_to_pickle(rgb_frames_dir, session_name, est_ppgs, rf_ppg, pickle_path, fs=30):
    """Save the generated PPG data and RF PPGs into a pickle file."""

    gt_ppg_path = os.path.join(rgb_frames_dir, session_name, 'rgbd_ppg.npy')
    gt_ppg = np.load(gt_ppg_path)

    print(f"Shape of est_ppg here is {est_ppgs.shape}")
    print(f"Shape of rf_ppg here is {rf_ppg.shape}")
    print(f"Shape of gt_ppg here is {gt_ppg.shape}")

    fusion_data = {
        'video_path': os.path.join(rgb_frames_dir, session_name),
        'est_ppgs': est_ppgs,
        'gt_ppgs': gt_ppg,  # As ground truth isn't available, we use estimated PPGs
        'rf_ppg': rf_ppg
    }

    # print(f"{len(fusion_data[0])}") #this HAS to match
    print(f"{len(fusion_data)}") #this HAS to match
    with open(pickle_path, 'wb') as f:
        pickle.dump([fusion_data], f)
    

    print(f"Fusion data saved to {pickle_path}")
    

def main(args):
    # Set the device
    if args.device is None:
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device(args.device)

    rf_demo = ['91_1']
    session_name_demo = ['v_91_1']

    # Load the RF model (RF_conv_decoder) for generating PPG from RF data
    rf_model = RF_conv_decoder().to(args.device)
    rf_ckpt_path = '/Users/jamesemilian/triage/equipleth/Camera_77GHzRadar_Plethysmography_2/best_pth/RF_IQ_Net/best.pth'  # Update this path
    rf_model.load_state_dict(torch.load(rf_ckpt_path, map_location=args.device))
    rf_demo_data =  gen_rf_preds(root_path=args.rf_dir, demo_files=rf_demo, model=rf_model, device=args.device)

    # Load the model for generating PPG from RGB frames
    rgb_model = CNN3D().to(args.device)
    rgb_ckpt_path = '/Users/jamesemilian/triage/equipleth/Camera_77GHzRadar_Plethysmography_2/best_pth/RGB_CNN3D/best.pth'  # Update this path
    rgb_model.load_state_dict(torch.load(rgb_ckpt_path, map_location=args.device))
    rgb_demo_data = gen_rgb_preds(root_dir=args.rgb_dir, session_name=session_name_demo, model=rgb_model, device=args.device)
    
    print(len(rf_demo_data))
    print(len(rgb_demo_data))
    # Combine the dictionaries
    total_ppg_list = []
    for i in range(len(rf_demo_data)):
        for j in range(len(rgb_demo_data)):
            if(rf_demo_data[i]['video_path'] == rgb_demo_data[j]['video_path'][2:] \
                                and 'est_ppgs' in rgb_demo_data[j]) and 'rf_ppg' in rf_demo_data[i]:
                rgb_demo_data[j]['rf_ppg'] = rf_demo_data[i]['rf_ppg']
                total_ppg_list.append(rgb_demo_data[j])

    # Extract RGB frames and generate estimated PPGs
    est_ppgs = rgb_demo_data[0]['est_ppg']

    # Process RF data to generate RF PPGs
    rf_ppg = rf_demo_data[0]['rf_ppg']

    # Save the generated data into a pickle file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pickle_file_path = f'/Users/jamesemilian/triage/equipleth/Camera_77GHzRadar_Plethysmography_2/fusion_dataset/gen_fusion_data_{timestamp}.pkl'
    session_name = 'v_91_1' #put this in args
    save_fusion_data_to_pickle(args.rgb_dir, session_name, est_ppgs, rf_ppg, pickle_file_path)

    session_name_demo = "/Users/jamesemilian/triage/equipleth/Camera_77GHzRadar_Plethysmography_2/rgb_files/v_1_1"
    # Load the saved pickle file and create a dataset for evaluation
    dataset_test = FusionRunDatasetObject(
        datapath=pickle_file_path,
        datafiles=session_name_demo,
        fft_resolution=48,
        desired_ppg_len=300,
        compute_fft=True
    )

    #what happens if I do this??
    # dataset_test = FusionEvalDatasetObject(
    #     datapath=pickle_file_path,
    #     datafiles=session_name_demo,
    #     fft_resolution=48,
    #     desired_ppg_len=300,
    #     compute_fft=True
    # )
    #same Dataset length: 0 error. Solve it.
    fusion_model = FusionModel(base_ppg_est_len=1024, rf_ppg_est_len=1024*5, out_len=1024).to(args.device)
    # Run forward pass using the fusion model
    _, _, session_name, hr_test, rr_test, waveforms = eval_fusion_model(dataset_test, fusion_model, method='both', device=args.device)

    # Visualize results
    est_wv_arr, gt_wv_arr, rgb_wv_arr, rf_wv_arr = waveforms
    plot_and_analyze_results(est_wv_arr[0], gt_wv_arr[0], rgb_wv_arr[0], rf_wv_arr[0])

def plot_and_analyze_results(ppg_fusion, ppg_gt, ppg_rgb, ppg_rf, fs=30):
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(ppg_fusion, label='Fusion Model PPG')
    plt.plot(ppg_gt, label='GT PPG')
    plt.plot(ppg_rgb, label='RGB Model PPG')
    plt.plot(ppg_rf, label='RF Model PPG')
    plt.title('PPG Predictions')
    plt.legend()

    _, fft_fusion_F, fft_fusion_Pxx = pulse_rate_from_power_spectral_density(ppg_fusion, 30, 45, 150)
    _, fft_rgb_F, fft_rgb_Pxx = pulse_rate_from_power_spectral_density(ppg_rgb, 30, 45, 150)
    _, fft_rf_F, fft_rf_Pxx = pulse_rate_from_power_spectral_density(ppg_rf, 30, 45, 150)
    _, fft_gt_F, fft_gt_Pxx = pulse_rate_from_power_spectral_density(ppg_gt, 30, 45, 150)
    # fft_fusion = np.fft.fft(ppg_fusion) #naive fft with no filtering
    # fft_rgb = np.fft.fft(ppg_rgb)
    # fft_rf = np.fft.fft(ppg_rf) 
    # fft_gt = np.fft.fft(ppg_gt)

    plt.subplot(2, 1, 2)
    plt.plot(fft_fusion_F, fft_fusion_Pxx, label='Fusion FFT')
    plt.plot(fft_gt_F, fft_gt_Pxx, label='GT FFT')
    plt.plot(fft_rgb_F, fft_rgb_Pxx, label='RGB FFT')
    plt.plot(fft_rf_F, fft_rf_Pxx, label='RF FFT')
    plt.title('FFT Outputs')
    plt.legend()

    plt.tight_layout()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'PPG_FFT_Analysis_Single_Sample{timestamp}.png')
    plt.show()

if __name__ == '__main__':
    args = parseArgs()
    main(args)
