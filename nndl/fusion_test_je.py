import os
import pickle
import numpy as np
import scipy.stats
import sklearn.metrics
import torch

from tqdm import tqdm

from rf import organizer as org
from rf.proc import create_fast_slow_matrix, find_range
from .errors import getErrors
from .utils import extract_video, pulse_rate_from_power_spectral_density


def eval_fusion_model(dataset_test, model, device = torch.device('cpu'), method = 'both'):
    model.eval()
    print(f"Method : {method}")
    mae_list = []
    mae_list_rr = []
    session_names = []
    hr_est_arr = []
    hr_gt_arr = []
    hr_rgb_arr = []
    hr_rf_arr = []
    rr_est_arr = [] #
    rr_gt_arr = []
    rr_rgb_arr = []
    rr_rf_arr = [] #
    est_wv_arr = []
    gt_wv_arr = []
    rgb_wv_arr = []
    rf_wv_arr = []
    print(f"Dataset length: {len(dataset_test)}")


    for i in range(len(dataset_test)):
        pred_ffts = []
        targ_ffts = []
        pred_rgbs = []
        pred_rfs  = []

        pred_ffts_rr = []
        targ_ffts_rr = []
        pred_rgbs_rr = []
        pred_rfs_rr  = []

        train_sig, gt_sig = dataset_test[i]
        print(f"Sample {i+1}/{len(dataset_test)}")
        print(f"train_sig keys: {train_sig.keys()}")
        print(f"train_sig est_ppgs shape: {train_sig['est_ppgs'].shape}")
        print(f"train_sig rf_ppg shape: {train_sig['rf_ppg'].shape}")
        print(f"gt_sig shape: {gt_sig.shape}")
        
        sess_name = dataset_test.all_combs[i][0]["video_path"]
        session_names.append(sess_name)
        
        train_sig['est_ppgs'] = torch.tensor(train_sig['est_ppgs']).type(torch.float32).to(device)
        train_sig['est_ppgs'] = torch.unsqueeze(train_sig['est_ppgs'], 0)
        train_sig['rf_ppg'] = torch.tensor(train_sig['rf_ppg']).type(torch.float32).to(device)
        train_sig['rf_ppg'] = torch.unsqueeze(train_sig['rf_ppg'], 0)

        gt_sig = torch.tensor(gt_sig).type(torch.float32).to(device)

        with torch.no_grad():
            if method.lower()  == 'rf':
                # Only RF, RGB is noise
                fft_ppg = model(torch.rand(torch.unsqueeze(train_sig['est_ppgs'], axis=0).shape).to(device), torch.unsqueeze(train_sig['rf_ppg'], axis=0))
            elif method.lower() == 'rgb':
                # Only RGB, RF is randn
                fft_ppg = model(torch.unsqueeze(train_sig['est_ppgs'], axis=0), torch.rand(torch.unsqueeze(train_sig['rf_ppg'], axis=0).shape).to(device))
            else:
                # Both RGB and RF
                input_rgb_ppg = torch.unsqueeze(train_sig['est_ppgs'], axis=0)
                input_rf_ppg = torch.unsqueeze(train_sig['rf_ppg'], axis=0)
                print(f"Inputs to the fusion model are of shape - RGB FFT: {input_rgb_ppg.shape} and RF FFT: {input_rf_ppg.shape}")
                fft_ppg = model(torch.unsqueeze(train_sig['est_ppgs'], axis=0), torch.unsqueeze(train_sig['rf_ppg'], axis=0))

        print(f"fft_ppg shape after model inference: {fft_ppg.shape}")
        
        # Reduce the dims
        fft_ppg = torch.squeeze(fft_ppg, 1)
        temp_fft = fft_ppg[0].detach().cpu().numpy()
        temp_fft = temp_fft-np.min(temp_fft)
        temp_fft = temp_fft/np.max(temp_fft) #normalized FFT inference from fusion model

        # Calculate iffts of original signals
        rppg_fft = train_sig['rppg_fft']
        rppg_mag = np.abs(rppg_fft)
        rppg_ang = np.angle(rppg_fft)
        # Replace magnitude with new spectrum
        lix = dataset_test.l_freq_idx 
        rix = dataset_test.u_freq_idx + 1
        roi = rppg_mag[lix:rix]
        temp_fft = temp_fft*np.max(roi)
        rppg_mag[lix:rix] = temp_fft
        rppg_mag[-rix+1:-lix+1] = np.flip(temp_fft)
        rppg_fft_est = rppg_mag*np.exp(1j*rppg_ang)

        rppg_est = np.real(np.fft.ifft(rppg_fft_est)) #PPG reconstruction from FFT from fusion model
        rppg_est = rppg_est[0:300] # The 300 is the same as desired_ppg_length given in the dataloader
        gt_est = np.real(np.fft.ifft(train_sig['gt_fft']))[0:300] #The 300 is the same as desired_ppg_length given in the dataloader

        # Re-normalize
        rppg_est = (rppg_est - np.mean(rppg_est)) / np.std(rppg_est)
        gt_est = (gt_est - np.mean(gt_est)) / np.std(gt_est)

        # pred_fft_value = pulse_rate_from_power_spectral_density(rppg_est, 30, 45, 150)
        pred_ffts.append(pulse_rate_from_power_spectral_density(rppg_est, 30, 45, 150)) #fusion model HR prediction
        # pred_ffts.append(pred_fft_value) #fusion model HR prediction
        # print(f'1: {pred_ffts}')
        targ_ffts.append(pulse_rate_from_power_spectral_density(gt_est, 30, 45, 150)) # GT HR
        pred_rgbs.append(pulse_rate_from_power_spectral_density(train_sig['rgb_true'], 30, 45, 150)) #RGB model HR prediction
        pred_rfs.append(pulse_rate_from_power_spectral_density(train_sig['rf_true'], 30, 45, 150)) #RF model HR prediction

        pred_ffts_rr.append(pulse_rate_from_power_spectral_density(rppg_est, 30, 4, 40)) #fusion model RR prediction
        targ_ffts_rr.append(pulse_rate_from_power_spectral_density(gt_est, 30, 4, 40)) # GT RR
        pred_rgbs_rr.append(pulse_rate_from_power_spectral_density(train_sig['rgb_true'], 30, 4, 40)) #RGB model RR prediction
        pred_rfs_rr.append(pulse_rate_from_power_spectral_density(train_sig['rf_true'], 30, 4, 40)) #RF model RR prediction

        #display GT and prediction values for debugging. (COMMENT OUT if not debugging)
        print(f"Current length of pred_ffts: {len(pred_ffts)}") 
        print(f'Index is: {i}')
        print(f"Predicted HR (Fusion): {pred_ffts[0]} bpm")
        print(f"Predicted RR (Fusion): {pred_ffts_rr[0]} bpm")
        print(f"Predicted HR (RGB): {pred_rgbs[0]} bpm")
        print(f"Predicted RR (RGB): {pred_rgbs_rr[0]} bpm")
        print(f"Predicted HR (RF): {pred_rfs[0]} bpm") 
        print(f"Predicted RR (RF): {pred_rfs_rr[0]} bpm") 
        print(f"Ground Truth HR: {targ_ffts[0]} bpm")
        print(f"Ground Truth RR: {targ_ffts_rr[0]} bpm")

        pred_ffts = np.array(pred_ffts)[:,np.newaxis]
        targ_ffts = np.array(targ_ffts)[:,np.newaxis]
        pred_rgbs = np.array(pred_rgbs)[:,np.newaxis]
        pred_rfs = np.array(pred_rfs)[:,np.newaxis]

        pred_ffts_rr = np.array(pred_ffts_rr)[:,np.newaxis]
        targ_ffts_rr = np.array(targ_ffts_rr)[:,np.newaxis]
        pred_rgbs_rr = np.array(pred_rgbs_rr)[:,np.newaxis]
        pred_rfs_rr = np.array(pred_rfs_rr)[:,np.newaxis]

        #why are we appending [1x1] arrays insted of just [1] value? Not sure.
        hr_est_arr.append(pred_ffts)
        hr_gt_arr.append(targ_ffts) 
        hr_rgb_arr.append(pred_rgbs) # array of RGB model HR predictions
        hr_rf_arr.append(pred_rfs) # array of RF model HR predictions

        rr_est_arr.append(pred_ffts_rr)
        rr_gt_arr.append(targ_ffts_rr)
        rr_rgb_arr.append(pred_rgbs_rr) # array of RGB model HR predictions
        rr_rf_arr.append(pred_rfs_rr) # array of RF model HR predictions

        _, MAE, _, _ = getErrors(pred_ffts, targ_ffts, PCC=False)
        _, MAE_rr, _, _ = getErrors(pred_ffts_rr, targ_ffts_rr, PCC=False) #adding this for RR MAE
        #can get the MAE for RGB and RF models as well.


        mae_list.append(MAE)
        mae_list_rr.append(MAE_rr)
        est_wv_arr.append(rppg_est) #ppg waveform estimated from fusion model
        gt_wv_arr.append(gt_est) #ppg waveform reconstructed from gt_fft
        rgb_wv_arr.append(train_sig['rgb_true']) #
        rf_wv_arr.append(train_sig['rf_true'])


    return np.array(mae_list), np.array(mae_list_rr), session_names, (hr_est_arr, hr_gt_arr), (rr_est_arr, rr_gt_arr), (est_wv_arr,gt_wv_arr, rgb_wv_arr, rf_wv_arr)
    # mae_list (hr), mae_list (rr), session_names (test), (hr_fusion_pred, hr_gt), (rr_fusion_pred, rr_gt), 
    # can also make it (hr_est_arr, hr_gt_arr, hr_rgb_arr, hr_rf_arr) to return the values from RGB and RF models as well. (but yep, there are separate files for that)
    # run this for now and see if the ppg plots look reasonable.

def run_fusion_model(dataset_test, model, device = torch.device('cpu'), method = 'both'):
    model.eval()
    print(f"Method : {method}")
    mae_list = []
    mae_list_rr = []
    session_names = []
    hr_est_arr = []
    # hr_gt_arr = []
    hr_rgb_arr = []
    hr_rf_arr = []
    rr_est_arr = [] #
    # rr_gt_arr = []
    rr_rgb_arr = []
    rr_rf_arr = [] #
    est_wv_arr = []
    # gt_wv_arr = []
    rgb_wv_arr = []
    rf_wv_arr = []
    print(f"Dataset length: {len(dataset_test)}")


    for i in range(len(dataset_test)):
        pred_ffts = []
        # targ_ffts = []
        pred_rgbs = []
        pred_rfs  = []

        pred_ffts_rr = []
        # targ_ffts_rr = []
        pred_rgbs_rr = []
        pred_rfs_rr  = []

        train_sig = dataset_test[i]
        print(f"Sample {i+1}/{len(dataset_test)}")
        print(f"train_sig keys: {train_sig.keys()}")
        print(f"train_sig est_ppgs shape: {train_sig['est_ppgs'].shape}")
        print(f"train_sig rf_ppg shape: {train_sig['rf_ppg'].shape}")
        # print(f"gt_sig shape: {gt_sig.shape}")
        
        sess_name = dataset_test.all_combs[i][0]["video_path"]
        session_names.append(sess_name)
        
        train_sig['est_ppgs'] = torch.tensor(train_sig['est_ppgs']).type(torch.float32).to(device)
        train_sig['est_ppgs'] = torch.unsqueeze(train_sig['est_ppgs'], 0)
        train_sig['rf_ppg'] = torch.tensor(train_sig['rf_ppg']).type(torch.float32).to(device)
        train_sig['rf_ppg'] = torch.unsqueeze(train_sig['rf_ppg'], 0)

        with torch.no_grad():
            if method.lower()  == 'rf':
                # Only RF, RGB is noise
                fft_ppg = model(torch.rand(torch.unsqueeze(train_sig['est_ppgs'], axis=0).shape).to(device), torch.unsqueeze(train_sig['rf_ppg'], axis=0))
            elif method.lower() == 'rgb':
                # Only RGB, RF is randn
                fft_ppg = model(torch.unsqueeze(train_sig['est_ppgs'], axis=0), torch.rand(torch.unsqueeze(train_sig['rf_ppg'], axis=0).shape).to(device))
            else:
                # Both RGB and RF
                input_rgb_ppg = torch.unsqueeze(train_sig['est_ppgs'], axis=0)
                input_rf_ppg = torch.unsqueeze(train_sig['rf_ppg'], axis=0)
                print(f"Inputs to the fusion model are of shape - RGB FFT: {input_rgb_ppg.shape} and RF FFT: {input_rf_ppg.shape}")
                fft_ppg = model(torch.unsqueeze(train_sig['est_ppgs'], axis=0), torch.unsqueeze(train_sig['rf_ppg'], axis=0))

        print(f"fft_ppg shape after model inference: {fft_ppg.shape}")
        
        # Reduce the dims
        fft_ppg = torch.squeeze(fft_ppg, 1)
        temp_fft = fft_ppg[0].detach().cpu().numpy()
        temp_fft = temp_fft-np.min(temp_fft)
        temp_fft = temp_fft/np.max(temp_fft)

        # Calculate iffts of original signals
        rppg_fft = train_sig['rppg_fft']
        rppg_mag = np.abs(rppg_fft)
        rppg_ang = np.angle(rppg_fft)
        # Replace magnitude with new spectrum
        lix = dataset_test.l_freq_idx 
        rix = dataset_test.u_freq_idx + 1
        roi = rppg_mag[lix:rix]
        temp_fft = temp_fft*np.max(roi)
        rppg_mag[lix:rix] = temp_fft
        rppg_mag[-rix+1:-lix+1] = np.flip(temp_fft)
        rppg_fft_est = rppg_mag*np.exp(1j*rppg_ang)

        rppg_est = np.real(np.fft.ifft(rppg_fft_est))
        rppg_est = rppg_est[0:300] # The 300 is the same as desired_ppg_length given in the dataloader
        # gt_est = np.real(np.fft.ifft(train_sig['gt_fft']))[0:300] #The 300 is the same as desired_ppg_length given in the dataloader

        # Re-normalize
        rppg_est = (rppg_est - np.mean(rppg_est)) / np.std(rppg_est)
        # gt_est = (gt_est - np.mean(gt_est)) / np.std(gt_est)

        # pred_fft_value = pulse_rate_from_power_spectral_density(rppg_est, 30, 45, 150)
        pred_ffts.append(pulse_rate_from_power_spectral_density(rppg_est, 30, 45, 150)) #fusion model HR prediction
        # pred_ffts.append(pred_fft_value) #fusion model HR prediction
        # print(f'1: {pred_ffts}')
        # targ_ffts.append(pulse_rate_from_power_spectral_density(gt_est, 30, 45, 150)) # GT HR
        pred_rgbs.append(pulse_rate_from_power_spectral_density(train_sig['rgb_true'], 30, 45, 150)) #RGB model HR prediction
        pred_rfs.append(pulse_rate_from_power_spectral_density(train_sig['rf_true'], 30, 45, 150)) #RF model HR prediction

        pred_ffts_rr.append(pulse_rate_from_power_spectral_density(rppg_est, 30, 4, 40)) #fusion model RR prediction
        # targ_ffts_rr.append(pulse_rate_from_power_spectral_density(gt_est, 30, 4, 40)) # GT RR
        pred_rgbs_rr.append(pulse_rate_from_power_spectral_density(train_sig['rgb_true'], 30, 4, 40)) #RGB model RR prediction
        pred_rfs_rr.append(pulse_rate_from_power_spectral_density(train_sig['rf_true'], 30, 4, 40)) #RF model RR prediction

        #display GT and prediction values for debugging. (COMMENT OUT if not debugging)
        print(f"Current length of pred_ffts: {len(pred_ffts)}") 
        print(f'Index is: {i}')
        print(f"Predicted HR (Fusion): {pred_ffts[0]} bpm")
        print(f"Predicted RR (Fusion): {pred_ffts_rr[0]} bpm")
        print(f"Predicted HR (RGB): {pred_rgbs[0]} bpm")
        print(f"Predicted RR (RGB): {pred_rgbs_rr[0]} bpm")
        print(f"Predicted HR (RF): {pred_rfs[0]} bpm") 
        print(f"Predicted RR (RF): {pred_rfs_rr[0]} bpm") 
        # print(f"Ground Truth HR: {targ_ffts[0]} bpm")
        # print(f"Ground Truth RR: {targ_ffts_rr[0]} bpm")

        pred_ffts = np.array(pred_ffts)[:,np.newaxis]
        # targ_ffts = np.array(targ_ffts)[:,np.newaxis]
        pred_rgbs = np.array(pred_rgbs)[:,np.newaxis]
        pred_rfs = np.array(pred_rfs)[:,np.newaxis]

        pred_ffts_rr = np.array(pred_ffts_rr)[:,np.newaxis]
        # targ_ffts_rr = np.array(targ_ffts_rr)[:,np.newaxis]
        pred_rgbs_rr = np.array(pred_rgbs_rr)[:,np.newaxis]
        pred_rfs_rr = np.array(pred_rfs_rr)[:,np.newaxis]

        #why are we appending [1x1] arrays insted of just [1] value? Not sure.
        hr_est_arr.append(pred_ffts)
        # hr_gt_arr.append(targ_ffts) 
        hr_rgb_arr.append(pred_rgbs) # array of RGB model HR predictions
        hr_rf_arr.append(pred_rfs) # array of RF model HR predictions

        rr_est_arr.append(pred_ffts_rr)
        # rr_gt_arr.append(targ_ffts_rr)
        rr_rgb_arr.append(pred_rgbs_rr) # array of RGB model HR predictions
        rr_rf_arr.append(pred_rfs_rr) # array of RF model HR predictions

        # _, MAE, _, _ = getErrors(pred_ffts, targ_ffts, PCC=False)
        # _, MAE_rr, _, _ = getErrors(pred_ffts_rr, targ_ffts_rr, PCC=False) #adding this for RR MAE
        #can get the MAE for RGB and RF models as well.


        # mae_list.append(MAE)
        # mae_list_rr.append(MAE_rr)
        est_wv_arr.append(rppg_est) #ppg waveform estimated from fusion model
        # gt_wv_arr.append(gt_est) #ppg waveform reconstructed from gt_fft
        rgb_wv_arr.append(train_sig['rgb_true']) #
        rf_wv_arr.append(train_sig['rf_true'])


    return 1, 1, session_names, (hr_est_arr, [1]), (rr_est_arr, [1]), (est_wv_arr,[1], rgb_wv_arr, rf_wv_arr)
    # mae_list (hr), mae_list (rr), session_names (test), (hr_fusion_pred, hr_gt), (rr_fusion_pred, rr_gt), 
    # can also make it (hr_est_arr, hr_gt_arr, hr_rgb_arr, hr_rf_arr) to return the values from RGB and RF models as well. (but yep, there are separate files for that)
    # run this for now and see if the ppg plots look reasonable.
