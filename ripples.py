import numpy as np
import os
from scipy.signal import butter, filtfilt, hilbert

def MapLFPs(path, nch, dtype=np.int16 ,order='F'):
        '''Returns a 2D numpy <memmap>-object to a binary file, which is indexable as [channel, sample].

        INPUT:
        - [path]:              <str> containing full path to binary-file
        - [nch]:               <int> number of channels in binary file
        - [dtype]=np.int16:    <numpy-type> of binary data points'''

        ## Calculate the total number of data points in the provided binary-file
        size = os.path.getsize(path)
        size = int(size/np.dtype(dtype).itemsize)

        ## Create and return the 2D memory-map object
        memMap = np.memmap(path, mode='r', dtype=dtype, order=order, shape=(nch, int(size/nch)))

        return memMap  

def downsample(data, original_fs, target_fs):
    downsample_factor = original_fs // target_fs
    if original_fs % target_fs != 0:
        raise ValueError("Original sampling rate must be an integer multiple of target sampling rate.")
    data = data[:,::downsample_factor]
    return data

def common_avg_ref(data):
    common_avg = np.mean(data, axis=0)  # shape: (n_timepoints,)
    data_car = data - common_avg[np.newaxis, :]  #
    return data_car

# Define filtering utility
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data, axis=1)

def detect_swr(data, ripple_fs, ripple_band=(80, 180), zscore_threshold=3):
    """
    Detects sharp-wave ripples in filtered LFP data.
    
    Parameters:
        data (np.ndarray): LFP data (channels x timepoints) at ripple sampling rate.
        ripple_fs (float): Sampling rate of the input data (Hz).
        ripple_band (tuple): Ripple frequency band (Hz).
        zscore_threshold (float): Threshold for SWR detection in z-score units.
    
    Returns:
        List of detected SWRs per channel: [(start_idx, end_idx), ...]
    """
    swr_events = []
    for ch in range(data.shape[0]):
        filtered = apply_filter(data[ch:ch+1, :], ripple_band[0], ripple_band[1], ripple_fs)
        envelope = np.abs(hilbert(filtered))  # Compute the analytic signal envelope
        zscored_envelope = (envelope - np.mean(envelope)) / np.std(envelope)
        above_thresh = np.where(zscored_envelope > zscore_threshold)[1]  # Get indices

        # Group into events
        events = []
        if len(above_thresh) > 0:
            event_start = above_thresh[0]
            for i in range(1, len(above_thresh)):
                if above_thresh[i] != above_thresh[i - 1] + 1:
                    event_end = above_thresh[i - 1]
                    events.append((event_start, event_end))
                    event_start = above_thresh[i]
            events.append((event_start, above_thresh[-1]))
        
        swr_events.append(events)
    return swr_events


def process_lfp(lfp_path, nch, original_fs=30000, target_fs=1000, ripple_band=(150, 250)):
    # Open memmap and reshape
    data = MapLFPs(lfp_path, nch)
    print(f"Original data shape: {data.shape}")

    # Downsample
    print("Downsampling...")
    downsampled_data = downsample(data, original_fs, target_fs)
    print(f"Downsampled data shape: {downsampled_data.shape}")
    
    print()
    car_data = common_avg_ref(downsampled_data)
    # Detect SWRs
    print("Detecting sharp-wave ripples...")
    swr_events = detect_swr(car_data, target_fs, ripple_band)
    
    return downsampled_data, swr_events