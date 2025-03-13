'''
Script to investigate EEG-ERD and EOG-HOV intuitiveness, defined as:
Time to reach the 50% of the maximum individual modulation after the cue presentation.
'''
import sys
import os
sys.path.append(os.getcwd())
import numpy as np
import pyxdf
import mne
from xdf2mne import stream2raw
from LSLStreamInfoInterface import find_channel_index, find_stream, get_parameters_from_xdf_stream
from matplotlib import pyplot as plt
from enums import Side
import seaborn as sns

def get_mean(eog_trials):
    if eog_trials.size == 0:
        return np.ones(eog_trials.shape[-1])*np.nan

    return eog_trials.mean(0)
def convert_to_numpy(name, eog):
    n_trials = len(eog)

    if n_trials == 0:
        return np.array([])

    target_len = int(round(EOG_trial_len * bci_srate))
    cutoff = min([len(e) for e in eog if len(e) >= target_len * 0.9])
    np_eog = np.stack([e[:cutoff] for e in eog if len(e) >= cutoff])

    if n_trials != np_eog.shape[0]:
        print("Attention: Removed ", n_trials - np_eog.shape[0],
              "trial(s) because more than 10% of the #samples were missing.")

    return np_eog

def extract_hov_trials():
    # separate eog data from left and right trials and
    # save trial directions and corresponding boolean indices
    eog_left = []
    eog_right = []

    trials = []
    for ix, cue in enumerate(cues[1:], start=1):
        if cue != cues[ix - 1]:
            if cue == cue_map['HOVLEFT']:
                trials += [(left, ix)]
                eog_left += [eog[(eog_times >= cue_times[ix]) &
                                 (eog_times <= cue_times[ix] + EOG_trial_len)]]
            elif cue == cue_map['HOVRIGHT']:
                trials += [(right, ix)]
                eog_right += [eog[(eog_times >= cue_times[ix]) &
                                  (eog_times <= cue_times[ix] + EOG_trial_len)]]

    # convert to numpy matrices
    eog_left = convert_to_numpy("left eog", eog_left)
    eog_right = convert_to_numpy("right eog", eog_right)

    return trials, eog_left, eog_right


participants_id = ['P01','P02','P03','P04','P05']

EOG_trial_len = 3  # length of trials in seconds
ERD_starts = []
EOG_starts = []
for p in participants_id:
    # Load EEG data from xdf file, containing all streams contined in the LSL cloud during the recording
    EEG_filepath = str("/Users/AnnalisaColucci/Desktop/Handy Rehab B:NE Data Analysis/Data/Group Analysis/EEG/" + p + "_EEG.xdf")
    # Load marker stream containing the experimental cues
    marker_stream, _ = pyxdf.load_xdf(EEG_filepath, dejitter_timestamps=True, select_streams=[{'name': 'TaskOutput'}])
    marker_stream = marker_stream[0]
    # Load preprocessed stream containing the data after online preprocessing
    # Containing preprocessed EEG signal from C3 and C4, and the EOG signal
    preprocessed_stream, _ = pyxdf.load_xdf(EEG_filepath, dejitter_timestamps=True, select_streams=[{'name': 'PreprocessedData'}])
    preprocessed_stream = preprocessed_stream[0]
    # Load feedback stream to get additional parameters, such as the laterality of the task, adjusted to each participant's paresis
    feedback_stream, _ = pyxdf.load_xdf(EEG_filepath, dejitter_timestamps=True, select_streams=[{'name': 'FeedbackStates'}])
    feedback_stream = feedback_stream[0]


    # Load the laterality of the hand for which motor imagery / motor attempt was executed
    # LEFT  (hand) -> brain signal from right hemisphere (C4)
    # RIGHT (hand) -> brain signal from left hemisphere (C3)
    Laterality = Side(get_parameters_from_xdf_stream(feedback_stream)['laterality'])

    # Extract preprocessed EEG data, convert into MNE raw_data object and assign the markers as annotations
    EEG_preprocessed_data, ev, ev_id = stream2raw(preprocessed_stream,marker_stream=marker_stream, marker_out=3)
    bci_srate = preprocessed_stream['info']['effective_srate'] # online sampling rate
    # Extract target EEG channels C3 and C4
    targets = {'prep': ['µC3', 'µC4']}
    prep_channel = find_channel_index(preprocessed_stream, targets['prep'])
    data_prep = EEG_preprocessed_data.pick_channels([ch for ch in targets['prep']])

    # Continue with the EEG controlateral to the paretic hand only
    # Calculate reference value (RV) on preprocessed data from Start till the end, and normalize the data based on the RV to ompute mu power relative changes
    if Laterality is Side.RIGHT:
        data_prepC3 = data_prep.copy().pick(targets['prep'][0])
        start = int(round(15 * bci_srate))  # Skip the first 15 seconds, the system is just starting
        rv = data_prepC3.get_data([0],start=start).mean()
        # Extract Close trials from the preprocessed data
        Close_processed_events = mne.events_from_annotations(data_prepC3, regexp="CLOSE")
        close_preprocessed_epochs = mne.Epochs(data_prepC3, Close_processed_events[0], tmin=0, tmax=5, baseline=None, preload=True)
        # Normalize close trials
        norm_close = (close_preprocessed_epochs._data /rv) - 1

    elif Laterality is Side.LEFT:
        data_prepC4 = data_prep.copy().pick(targets['prep'][1])
        start = int(round(15 * bci_srate))  # Skip the first 15 seconds, the system is just starting
        rv = data_prepC4.get_data([0],start=start).mean()
        # Extract Close trials from the preprocessed data
        Close_processed_events = mne.events_from_annotations(data_prepC4, regexp="CLOSE")
        close_preprocessed_epochs = mne.Epochs (data_prepC4,Close_processed_events[0], tmin=0,tmax=5, baseline=None, preload=True)
        # Normalize close trials
        norm_close = (close_preprocessed_epochs._data/rv) - 1

    #Compute ERD Time To Initialize
    ERD_threshold = np.min(norm_close.copy().mean(axis=(0,1)))/2
    ERD_start = (np.argwhere(norm_close.mean(axis=(0,1)) < ERD_threshold)[0])/bci_srate
    ERD_starts.append(ERD_start[0])

    #Load EOG data from xdf file, containing all streams contined in the LSL cloud during the recording
    EOG_filepath = str("/Users/AnnalisaColucci/Desktop/Handy Rehab B:NE Data Analysis/Data/Group Analysis/EOG/" + p + "_EOG.xdf")
    EOG_marker_stream, _ = pyxdf.load_xdf(EOG_filepath, dejitter_timestamps=True, select_streams=[{'name': 'TaskOutput'}])
    EOG_marker_stream = EOG_marker_stream[0]

    EOG_preprocessed_stream, _ = pyxdf.load_xdf(EOG_filepath, dejitter_timestamps=True, select_streams=[{'name': 'PreprocessedData'}])
    EOG_preprocessed_stream = EOG_preprocessed_stream[0]

    EOG_preprocessed_data, ev, ev_id = stream2raw(EOG_preprocessed_stream,marker_stream=EOG_marker_stream, marker_out=0)

    # get cue mappings
    cue_map = dict(zip(
        EOG_marker_stream['info']['desc'][0]['mappings'][0]['cues'][0].keys(),
        map(lambda x: int(x[0]),
            EOG_marker_stream['info']['desc'][0]['mappings'][0]['cues'][0].values())
    ))

    # extract time series and time stamps
    eog = EOG_preprocessed_stream['time_series'][:, 0].squeeze()
    eog_times = EOG_preprocessed_stream['time_stamps']
    cues = np.array(EOG_marker_stream['time_series'])[:, 0].squeeze().astype(int)
    cue_times = EOG_marker_stream['time_stamps']

    # start timestamps with 0
    start_time = min(cue_times[0], eog_times[0])
    eog_times -= start_time
    cue_times -= start_time

    # extract update sampling frequency
    BCI_srate = round(EOG_preprocessed_stream['info']['effective_srate'])

    # define cue sides
    left = int(EOG_marker_stream['info']['desc'][0]['mappings'][0]['cues'][0]['HOVLEFT'][0])
    right = int(EOG_marker_stream['info']['desc'][0]['mappings'][0]['cues'][0]['HOVRIGHT'][0])

    (trials, eog_left, eog_right) = extract_hov_trials()
    print(len(trials))

    if Laterality is Side.RIGHT:
        eog_mean = eog_left.copy().mean(axis=(0,1))
        eog_norm = (eog_left/eog_mean)-1

    elif Laterality is Side.LEFT:
        eog_mean = eog_right.copy().mean(axis=(0,1))
        eog_norm = (eog_right/eog_mean)-1

    #Compute EOG Time To Initialize
    EOG_threshold = np.max(eog_norm.mean(axis=0))/2
    EOG_start= (np.argwhere(eog_norm.mean(axis=0) > EOG_threshold)[0])/bci_srate
    EOG_starts.append(EOG_start[0])

print("ERD mean "+ str(np.array(ERD_starts).mean()))
print("ERD std: "+ str(np.std(np.array(ERD_starts))))
print("EOG mean "+ str(np.array(EOG_starts).mean()))
print("EOG std: "+ str(np.std(np.array(EOG_starts))))


# Plot Time to Initialize
plt.figure(figsize=[7,6])
palette =['green','red']
sns.set(style='whitegrid')
sns.boxplot([ERD_starts,EOG_starts],palette=palette,width=0.6, linewidth=0.5)
sns.stripplot([ERD_starts,EOG_starts],palette=palette, edgecolor='black', linewidth=1)
plt.xticks([0,1],['ERD', 'HOV'],fontsize=25)
plt.ylim([0,3.5])
plt.yticks(fontsize=15)
plt.axhline(3,linestyle='-.',color='grey', linewidth=0.7)
plt.ylabel('Time (s)', fontsize=25)
plt.title("Time To Initialize", fontsize=27)
plt.grid(False)
#plt.savefig("TimeToInitializeV2.svg", format="svg")
plt.show()
print("END")