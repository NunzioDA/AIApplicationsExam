from matplotlib import pyplot as plt
from sklearn.calibration import LabelEncoder
import wfdb
import numpy as np
import os

def load():
    """
    Loads the MIT-BIH Arrhythmia dataset records, combines them into NumPy arrays,
    filters the targets, saves the data and targets to .npy files if they don't exist,
    and returns the dataset and targets.

    Args:
        data_filename (str): The filename for saving the data array.
        targets_filename (str): The filename for saving the targets array.

    Returns:
        tuple: A tuple containing two NumPy arrays: the filtered data and the filtered targets.
               If the files already exist, returns the loaded content.
    """
    dataset_dir = "mit-bih-arrhythmia"
    os.makedirs(dataset_dir, exist_ok=True)
    data_filename=f"{dataset_dir}/mitbih_data.npy"
    symbols_filename=f"{dataset_dir}/mitbih_symbols.npy"
    targets_filename=f"{dataset_dir}/mitbih_targets.npy"


    if os.path.exists(data_filename) and os.path.exists(symbols_filename):# and os.path.exists(targets_filename) :
        print("Loading cached data...")
        data = np.load(data_filename)
        #targets = np.load(targets_filename)
        symbols = np.load(symbols_filename)
        return data, symbols#, targets
    else:
        print("Downloading and processing MIT-BIH Arrhythmia dataset...")
        # List of MIT-BIH Arrhythmia records to load
        record_names = ['100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
                        '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
                        '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
                        '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
                        '222', '223', '228', '230', '231', '232', '233', '234']

        all_data = []
        all_targets = []

        window_size = 360  
        half_window = window_size // 2

        
        valid_beats = ['N', 'L', 'R', 'A', 'a', 'V', 'F', 'e', 'j', 'E', 'S']

        for record_name in record_names:
            try:
                print(f"Downloading {record_name}...")
                record = wfdb.rdrecord(record_name, pn_dir='mitdb')
                annotation = wfdb.rdann(record_name, 'atr', pn_dir='mitdb')
                signal = record.p_signal[:, 0] 

                for i, sample in enumerate(annotation.sample):
                    symbol = annotation.symbol[i]
                    if symbol in valid_beats:
                        start = sample - half_window
                        end = sample + half_window
                        if start >= 0 and end < len(signal):
                            beat_segment = signal[start:end]
                            all_data.append(beat_segment)
                            all_targets.append(symbol)
            except FileNotFoundError:
                print(f"Error: Could not find record {record_name}")

        all_data = np.array(all_data)
        all_targets = np.array(all_targets)

        # label_encoder = LabelEncoder()
        # integer_encoded = label_encoder.fit_transform(all_targets)
        # one_hot = to_categorical(integer_encoded)

        np.save(data_filename, all_data)
        np.save(symbols_filename, all_targets)
        #np.save(targets_filename, one_hot)

        return all_data, all_targets#, one_hot

