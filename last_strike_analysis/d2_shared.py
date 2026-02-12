import numpy as np
import pandas as pd
import glob
import os
import gc
import seaborn as sns
from scipy.stats import zscore
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.family"] = "Arial"
rcParams["pdf.fonttype"] = 42      # Ensures text is saved as real characters
rcParams["ps.fonttype"] = 42
rcParams["svg.fonttype"] = "none"  # Keep SVG text selectable
import statsmodels.api as sm

roach_hunters = ["RCC00036", "RCC00040", "RCC00042", "RCC00043", "RCC00052", "RCC00059", "RCC00083"]
non_hunters   = ["RCC00041", "RCC00045", "RCC00053", "RCC00058"]

sorting_thsd_dict = {
    "RCC00036" : 5, "RCC00045" : 5,  "RCC00040" : 5, "RCC00042" : 5, "RCC00041" : 5,
    "RCC00043" : 4.5, "RCC00052": 5, "RCC00053" : 4.5, "RCC00058": 5, "RCC00059": 5,
    "RCC00083" : 5.5, "RCC00094": 5.5,
}

def to_timedelta(t_str):
    return pd.to_timedelta(t_str)

def compute_wall_time(row):
    filename = row["Video_File"]
    datetime_str = filename.split('-')[1]  # '20250522T103737'
    base_dt = datetime.strptime(datetime_str, "%Y%m%dT%H%M%S")
    delta = pd.to_timedelta(row["start_hunt"])
    return base_dt + delta

def load_in_prey_capture_record(animal):
    preyCaptureRecord = pd.read_csv(f"{animal}_PC_record.csv")
    preyCaptureRecord["animal"] = animal
    preyCaptureRecord["group"] = preyCaptureRecord["animal"].apply(lambda x: "hunter" if x in roach_hunters else "non_hunter")
    preyCaptureRecord["total_trial"] = ( preyCaptureRecord["trial"] + (preyCaptureRecord["day"] - 1) * 6 ).clip(lower=0)
    preyCaptureRecord["capture_time_sec"] = pd.to_timedelta(preyCaptureRecord["capture_time"]).dt.total_seconds()
    preyCaptureRecord["start_wall_time"] = preyCaptureRecord.apply(compute_wall_time, axis=1)
    preyCaptureRecord["end_wall_time"] = preyCaptureRecord["start_wall_time"] + pd.to_timedelta(preyCaptureRecord["capture_time_sec"], unit="s")
    return preyCaptureRecord

def load_animal_neural_data_df(animal, sorting_thsd_dict, sorted_data_format):
    ''' Load all sorted neuron object files, preparing to merge it to preyCaptureRecord df'''
    sorted_loc = f"/media/HlabShare/Clustering_Data/{animal}/block_4hr"
    sorted_files = glob.glob(f"{sorted_loc}/*thsd_{sorting_thsd_dict[animal]}/*/probe1/co/*{sorted_data_format}")
    print(f"I got {len(sorted_files)} sorted neuron objects for animal: {animal}")
    index = []
    for path in sorted_files:
        s_time, e_time = get_time_range(path, sorted_data_format)
        clean_path = path.replace(".npy", "_clean.npy")
        if os.path.exists(clean_path):
            # print(f"loading in cleaned neuron obj: {clean_path}")
            path = clean_path
        index.append({
            "filepath": path,
            "start": s_time,
            "end": e_time
        })
    sorted_index_df = pd.DataFrame(index)
    sorted_index_df = sorted_index_df.sort_values("start").reset_index(drop=True)
    return sorted_index_df

def find_matching_sorted_file(timestamp, sorted_index_df):
    '''Match each preyCaptureRecord row to the sorted file'''
    match = sorted_index_df[
        (sorted_index_df["start"] <= timestamp) & (timestamp < sorted_index_df["end"])
    ]
    if len(match) == 0:
        return pd.Series([None, None, None])
    row = match.iloc[0]
    file = row["filepath"]
    short_file = f"hour {Path(file).parent.parent.parent.name}=>{Path(file).name}"
    offset = (timestamp - row["start"]).total_seconds()
    return pd.Series([file, short_file, offset])

def get_binned_cross_spikes(sorted_file, start_time, end_time, bin_size_ms, # milliseconds
                            neuron_quality_thsd = 2, verbose=1,
                            return_num_neurons = False, only_RSU = False):
    """
    Computes binned spike counts in a time window.
    2 get_binned_cross_spikes() functions may be used to handle to edge case where the time period spans across 2 sorting blocks.
    
    Parameters:
    -----------
    sorted_file : neurons_group0_file (sorted neurons)
    start_time : when to start the binning (in seconds)
    end_time : when to end, -1 means the end of the file

    Returns:
    --------
    total_hist :  Binned spike counts (sum across neurons).
    """ 
    all_neurons = np.load(sorted_file, allow_pickle=True)
    neurons = [n for n in all_neurons if n.quality <= neuron_quality_thsd]
    # for n in neurons:
    #     n.remove_large_amplitude_spikes(2, lstd_deviation=True, start=False, end=False, lplot=False) 
    RSUs = [n for n in neurons if n.cell_type == 'RSU'] # Regular Spking Units, not fast spiking cells
    if verbose == True:
        print(f"Total {len(neurons)} neurons, RSU: {len(RSUs)} for {Path(sorted_file).name}")
    if only_RSU:
        neurons = RSUs
    s_dur = neurons[0].end_time - neurons[0].start_time # sorting duration

    if end_time == -1:
        end_time = s_dur
    if end_time > s_dur:
        print(f"Sorting actually ends at {s_dur} seconds. Modifying end_time")
        end_time = s_dur
    if start_time > end_time:
        raise ValueError("start_time greater than end_time")

    num_bins = int((end_time - start_time) / (bin_size_ms / 1000))
    total_hist = np.zeros(num_bins)
    
    for i, n in enumerate(neurons):
        spike_times = np.asarray(n.spike_time_sec)
        if spike_times.size == 0:
            continue  # skip empty neurons
        spikes = spike_times[(spike_times >= start_time) & (spike_times < end_time)]
        hist, _ = np.histogram(spikes, bins=num_bins, range=(start_time, end_time)) # bin spikes
        total_hist += hist
        if verbose >= 2:
            print(f"Neuron {i} (clust_idx={n.clust_idx}): {len(spikes)} spikes")
    if verbose:
        print(f"Total spikes across all neurons: {int(total_hist.sum())}")
        print(f"Binned into {num_bins} bins of {bin_size_ms} ms")
    if return_num_neurons:
        return total_hist, len(neurons)
    else:
        return total_hist

def sm_d2(sig, order):
    coefs, _ = sm.regression.yule_walker(sig, order=order,method="mle")
    return (1-np.sum(coefs))/np.sqrt(order)

def safe_d2_calc(arr, remove_zero = False, d2_order = 5):
    '''
    selection of the d2 order and the bin size is based on this from Sam Sooter's paper:
    "Cortex deviates from criticality during action and deep sleep: a temporal renormalization group approach"
    The primary results here (Fig 2) were robust over a range of model orders (5 to 20) and time bin choices (10 to 160 ms) (Supp Fig S1).
    '''
    if arr is None or isinstance(arr, float) and np.isnan(arr):
        return None
    if remove_zero:
        arr = arr[arr > 0]
    z = zscore(arr)
    d2v1 = sm_d2(z, d2_order)
    return d2v1

def get_time_range(sorted_filename, sorted_data_format): # H_2025-05-19_23-28-10_2025-05-20_03-23-10_neurons_group0.npy'
    time_stamp = Path(sorted_filename).name.replace(sorted_data_format, "")
    _, start_date, start_time, end_date, end_time, _ = time_stamp.split("_")
    s_datetime = datetime.strptime(f"{start_date}-{start_time}", "%Y-%m-%d-%H-%M-%S")
    e_datetime = datetime.strptime(f"{end_date}-{end_time}", "%Y-%m-%d-%H-%M-%S")
    return s_datetime, e_datetime

start_at_next_file = [ # the field "start_sorted_file" will take the value of hunt_sorted_file
    ("start_sorted_file", "hunt_sorted_file"),
    ("start_offset_sec", 0.0)
]
end_by_previous_file = [ # the field "hunt_sorted_file" will take the value of start_sorted_file
    ("hunt_sorted_file", "start_sorted_file"),
    ("hunt_offset_sec", -1)
]

# because some recording during hunt is noisy, in case where I have sorting data for part of the d2 calculation block, I will just grab whetever data's that's available to me
def apply_manual_fixes(df, conditions_and_fixes): 
    for key, fix_list in conditions_and_fixes.items():
        animal, start_hunt = key.split("_")
        condition = ( (df["animal"] == animal) & (df["start_hunt"] == start_hunt) )
        for column_name, val in fix_list:
            if val in ["start_sorted_file", "hunt_sorted_file"]:
                val = (df.loc[condition, val])
            df.loc[condition, column_name] = val
    return df
