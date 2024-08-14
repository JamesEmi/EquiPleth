import os
import pickle
import numpy as np 
import imageio
import scipy.signal as sig
from torch.utils.data import Dataset

import rf.organizer as org
from rf.proc import create_fast_slow_matrix, find_range


class FusionEvalDatasetObject(Dataset):
    def __init__(self, datapath, datafiles, \
                    compute_fft=True, fs=30, l_freq_bpm=45, u_freq_bpm=180, \
                        desired_ppg_len=None, fft_resolution = 1, num_static_samples=7, window_rf=False, rf_window_size=5) -> None:        
        # There is an offset in the dataset between the captured video and GT
        self.ppg_offset = 25
        #Data structure for videos
        self.datapath = datapath
        self.datafiles = datafiles
        
        self.desired_ppg_len = desired_ppg_len
        self.compute_fft = compute_fft
        
        self.fs = fs
        self.l_freq_bpm = l_freq_bpm
        self.u_freq_bpm = u_freq_bpm

        self.window_rf = window_rf
        self.fft_resolution = fft_resolution
        self.rf_window_size = rf_window_size

        # Load the data from the pickle file
        with open(datapath, 'rb') as f:
            pickle_data = pickle.load(f)
        
        # Is any of the 4 keys (video path, estimated ppg from rgb, ground truth ppg, ppg from rf) is missing, we drop that point
        self.usable_data = []
        for data_pt in pickle_data:
            if data_pt['video_path'] in self.datafiles:
                if len(data_pt) != 4:
                    # self.usable_data.remove(data_pt)
                    print(f"{data_pt['video_path']} is dropped")
                    continue
                self.usable_data.append(data_pt)

        
        # If we want to use smaller window of the signals rather than the whole signal itself
        self.all_combs = []
        if self.desired_ppg_len is not None:
            self.num_static_samples = num_static_samples
            for data_pt in self.usable_data:
                # TODO crosscheck this and pass as a param
                static_idxs = np.array([0,128,256,384,512])

                for idx in static_idxs:
                    self.all_combs.append((data_pt, idx))
            seq_len = self.desired_ppg_len*self.fft_resolution
        else:
            for data_pt in self.usable_data:
                self.all_combs.append((data_pt, None))
                seq_len = len(data_pt['gt_ppgs'])*self.fft_resolution
        print(f"Dataset Ready. There are {self.__len__()} samples")

        freqs_bpm = np.fft.fftfreq(seq_len, d=1/self.fs) * 60
        self.l_freq_idx = np.argmin(np.abs(freqs_bpm - self.l_freq_bpm))
        self.u_freq_idx = np.argmin(np.abs(freqs_bpm - self.u_freq_bpm))
        print(self.l_freq_idx, self.u_freq_idx)
        print(freqs_bpm[self.l_freq_idx], freqs_bpm[self.u_freq_idx])
        assert self.l_freq_idx < self.u_freq_idx
        
            
    def __len__(self):
        return len(self.all_combs)

    def __getitem__(self, idx):

        dict_item, start_idx = self.all_combs[idx]
        # dict_keys(['video_path', 'est_ppgs', 'gt_ppgs', 'rf_ppg'])
        # Get the ppg data of the rgb, gt and rf
        item = {'est_ppgs':dict_item['est_ppgs'], 'rf_ppg':dict_item['rf_ppg']}
        item_sig = dict_item['gt_ppgs']
        if self.desired_ppg_len is not None:
            assert start_idx is not None
            item_sig = item_sig[start_idx+self.ppg_offset:start_idx+self.ppg_offset+self.desired_ppg_len]
            item['est_ppgs'] = item['est_ppgs'][start_idx:start_idx+self.desired_ppg_len]
            item['rf_ppg'] = item['rf_ppg'][start_idx:start_idx+self.desired_ppg_len]
        
        item_sig = (item_sig - np.mean(item_sig)) / np.std(item_sig)
        item['est_ppgs'] = (item['est_ppgs'] - np.mean(item['est_ppgs'])) / np.std(item['est_ppgs'])
        item['rf_ppg'] = (item['rf_ppg'] - np.mean(item['rf_ppg'], axis = 0)) / np.std(item['rf_ppg'], axis = 0)

        if self.compute_fft:
            n_curr = len(item_sig) * self.fft_resolution
            fft_gt  = np.abs(np.fft.fft(item_sig, n=int(n_curr), axis=0))
            fft_gt = fft_gt / np.max(fft_gt, axis=0)
            
            fft_est = np.abs(np.fft.fft(item['est_ppgs'], n=int(n_curr), axis=0))
            fft_est = fft_est / np.max(fft_est, axis=0)
            fft_est = fft_est[self.l_freq_idx : self.u_freq_idx + 1]

            fft_rf  = np.abs(np.fft.fft(item['rf_ppg'], n=int(n_curr), axis=0))
            fft_rf = fft_rf[self.l_freq_idx : self.u_freq_idx + 1]

            #Get full ffts
            rppg_fft = np.fft.fft(item['est_ppgs'], n=int(n_curr), axis=0)
            rf_fft = np.fft.fft(item['rf_ppg'], n=int(n_curr), axis=0)
            gt_fft = np.fft.fft(item_sig, n=int(n_curr), axis=0)

            if(self.window_rf):
                center_idx = np.argmax(fft_est)
                window_size = self.rf_window_size
                if(center_idx - window_size <= 0):
                    center_idx = window_size + 1
                elif(center_idx + window_size + 1 >= len(fft_est)):
                    center_idx = len(fft_est) - window_size - 1
                mask = np.zeros_like(fft_rf)
                mask[center_idx-window_size:center_idx+window_size+1,:] = 1
                fft_rf = np.multiply(fft_rf, mask)
                fft_rf = fft_rf / np.max(fft_rf)
            else:
                fft_rf = fft_rf / np.max(fft_rf, axis=0)
            
            return {'est_ppgs':fft_est, 'rf_ppg':fft_rf, 'rppg_fft':rppg_fft, 'rf_fft':rf_fft, 'gt_fft':gt_fft, 'rgb_true': item['est_ppgs'], 'rf_true': item['rf_ppg'], 'start_idx': start_idx}, fft_gt[self.l_freq_idx : self.u_freq_idx + 1]
        # If compute_fft is True; FFTs are returned, along with the original PPG waveforms too.
        else:
            item_sig         = self.lowPassFilter(item_sig)
            item['est_ppgs'] = self.lowPassFilter(item['est_ppgs'])
            for i in range(item['rf_ppg'].shape[1]):
                item['rf_ppg'][:,i]   = self.lowPassFilter(item['rf_ppg'][:,i])

            item_sig = (item_sig - np.mean(item_sig)) / np.std(item_sig)
            item['est_ppgs'] = (item['est_ppgs'] - np.mean(item['est_ppgs'])) / np.std(item['est_ppgs'])
            item['rf_ppg'] = (item['rf_ppg'] - np.mean(item['rf_ppg'], axis = 0)) / np.std(item['rf_ppg'], axis = 0)

            return item, np.array(item_sig)
        
class FusionRunDatasetObject(Dataset):
    def __init__(self, datapath, datafiles, \
                    compute_fft=True, fs=30, l_freq_bpm=45, u_freq_bpm=180, \
                        desired_ppg_len=None, fft_resolution = 1, num_static_samples=7, window_rf=False, rf_window_size=5) -> None:        
        # There is an offset in the dataset between the captured video and GT
        self.ppg_offset = 25
        #Data structure for videos
        self.datapath = datapath
        self.datafiles = datafiles
        
        self.desired_ppg_len = desired_ppg_len
        self.compute_fft = compute_fft
        
        self.fs = fs
        self.l_freq_bpm = l_freq_bpm
        self.u_freq_bpm = u_freq_bpm

        self.window_rf = window_rf
        self.fft_resolution = fft_resolution
        self.rf_window_size = rf_window_size

        # Load the data from the pickle file
        with open(datapath, 'rb') as f:
            pickle_data = pickle.load(f)
        
        # Is any of the 4 keys (video path, estimated ppg from rgb, ground truth ppg, ppg from rf) is missing, we drop that point
        self.usable_data = []
        for data_pt in pickle_data:
            # print(data_pt)
            # print(data_pt[0].keys())
            # data_pt = data_pt[0]
            print(f'Data_pt keys are: {data_pt.keys()}')
            # if data_pt['video_path'] in self.datafiles:
            if len(data_pt) != 4:
                # self.usable_data.remove(data_pt)
                print(f"{data_pt['video_path']} is dropped")
                print(f"Data does not have all 4 keys, check your pkl file/loading")
                continue
            
            self.usable_data.append(data_pt)
            len(f'Length of usable_data is {self.usable_data}')

        
        # If we want to use smaller window of the signals rather than the whole signal itself
        self.all_combs = []
        print(f"ppg len is :{self.desired_ppg_len}")
        if self.desired_ppg_len is not None:
            print(f"HERE")
            self.num_static_samples = num_static_samples
            for data_pt in self.usable_data:
                # TODO crosscheck this and pass as a param
                print(f'data_pt here is: {data_pt}')
                # static_idxs = np.array([0,128,256,384,512]) #this is hardcoded to lead to 5 combs
                static_idxs = np.array([0])

                for idx in static_idxs:
                    self.all_combs.append((data_pt, idx))
                print(f"Length of all_combs is:{len(self.all_combs)}")
            seq_len = self.desired_ppg_len*self.fft_resolution
        else:
            for data_pt in self.usable_data:
                self.all_combs.append((data_pt, None))
                seq_len = len(data_pt['gt_ppgs'])*self.fft_resolution
        print(f"Dataset Ready. There are {self.__len__()} samples")

        freqs_bpm = np.fft.fftfreq(seq_len, d=1/self.fs) * 60
        self.l_freq_idx = np.argmin(np.abs(freqs_bpm - self.l_freq_bpm))
        self.u_freq_idx = np.argmin(np.abs(freqs_bpm - self.u_freq_bpm))
        print(self.l_freq_idx, self.u_freq_idx)
        print(freqs_bpm[self.l_freq_idx], freqs_bpm[self.u_freq_idx])
        assert self.l_freq_idx < self.u_freq_idx
        
            
    def __len__(self):
        return len(self.all_combs)

    def __getitem__(self, idx):

        dict_item, start_idx = self.all_combs[idx]
        # dict_keys(['video_path', 'est_ppgs', 'gt_ppgs', 'rf_ppg'])
        # Get the ppg data of the rgb, gt and rf
        item = {'est_ppgs':dict_item['est_ppgs'], 'rf_ppg':dict_item['rf_ppg']}
        item_sig = dict_item['gt_ppgs']
        if self.desired_ppg_len is not None:
            assert start_idx is not None
            item_sig = item_sig[start_idx+self.ppg_offset:start_idx+self.ppg_offset+self.desired_ppg_len]
            item['est_ppgs'] = item['est_ppgs'][start_idx:start_idx+self.desired_ppg_len]
            item['rf_ppg'] = item['rf_ppg'][start_idx:start_idx+self.desired_ppg_len]
        
        item_sig = (item_sig - np.mean(item_sig)) / np.std(item_sig)
        item['est_ppgs'] = (item['est_ppgs'] - np.mean(item['est_ppgs'])) / np.std(item['est_ppgs'])
        item['rf_ppg'] = (item['rf_ppg'] - np.mean(item['rf_ppg'], axis = 0)) / np.std(item['rf_ppg'], axis = 0)

        if self.compute_fft:
            n_curr = len(item_sig) * self.fft_resolution
            fft_gt  = np.abs(np.fft.fft(item_sig, n=int(n_curr), axis=0)) #does this make sense??
            fft_gt = fft_gt / np.max(fft_gt, axis=0)
            
            fft_est = np.abs(np.fft.fft(item['est_ppgs'], n=int(n_curr), axis=0))
            fft_est = fft_est / np.max(fft_est, axis=0)
            fft_est = fft_est[self.l_freq_idx : self.u_freq_idx + 1]

            fft_rf  = np.abs(np.fft.fft(item['rf_ppg'], n=int(n_curr), axis=0))
            fft_rf = fft_rf[self.l_freq_idx : self.u_freq_idx + 1]

            #Get full ffts
            rppg_fft = np.fft.fft(item['est_ppgs'], n=int(n_curr), axis=0)
            rf_fft = np.fft.fft(item['rf_ppg'], n=int(n_curr), axis=0)
            gt_fft = np.fft.fft(item_sig, n=int(n_curr), axis=0)

            if(self.window_rf):
                center_idx = np.argmax(fft_est)
                window_size = self.rf_window_size
                if(center_idx - window_size <= 0):
                    center_idx = window_size + 1
                elif(center_idx + window_size + 1 >= len(fft_est)):
                    center_idx = len(fft_est) - window_size - 1
                mask = np.zeros_like(fft_rf)
                mask[center_idx-window_size:center_idx+window_size+1,:] = 1
                fft_rf = np.multiply(fft_rf, mask)
                fft_rf = fft_rf / np.max(fft_rf)
            else:
                fft_rf = fft_rf / np.max(fft_rf, axis=0)
            
            return {'est_ppgs':fft_est, 'rf_ppg':fft_rf, 'rppg_fft':rppg_fft, 'rf_fft':rf_fft, 'gt_fft':gt_fft, 'rgb_true': item['est_ppgs'], 'rf_true': item['rf_ppg'], 'start_idx': start_idx}, fft_gt[self.l_freq_idx : self.u_freq_idx + 1]
        # If compute_fft is True; FFTs are returned, along with the original PPG waveforms too.
        else:
            item_sig         = self.lowPassFilter(item_sig)
            item['est_ppgs'] = self.lowPassFilter(item['est_ppgs'])
            for i in range(item['rf_ppg'].shape[1]):
                item['rf_ppg'][:,i]   = self.lowPassFilter(item['rf_ppg'][:,i])

            item_sig = (item_sig - np.mean(item_sig)) / np.std(item_sig)
            item['est_ppgs'] = (item['est_ppgs'] - np.mean(item['est_ppgs'])) / np.std(item['est_ppgs'])
            item['rf_ppg'] = (item['rf_ppg'] - np.mean(item['rf_ppg'], axis = 0)) / np.std(item['rf_ppg'], axis = 0)

            return item, np.array(item_sig)

    def lowPassFilter(self, BVP, butter_order=4):
        [b, a] = sig.butter(butter_order, [self.l_freq_bpm/60, self.u_freq_bpm/60], btype='bandpass', fs = self.fs)
        filtered_BVP = sig.filtfilt(b, a, np.double(BVP))
        return filtered_BVP
