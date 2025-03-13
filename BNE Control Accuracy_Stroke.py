'''
Script to investigate EEG-ERD Control Accuracy, defined as:
participants ability to modulate their SMR activity below the 75 percentile of SMR activity during motor trials.
Participants have a reliable control when they can reach the 75 percentile during motor task (True Positive), but not at rest (True Negative).
[between seconds 1-5]

EOG-HOV Reliability is defined as:
modulation above the 75 percentile of ocular signal within each trial.
'''

import sys
import os
sys.path.append(os.getcwd())
import numpy as np
import pyxdf
import mne
from xdf2mne import stream2raw
from LSLStreamInfoInterface import find_channel_index, get_parameters_from_xdf_stream
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

    target_len = int(round(trial_len * bci_freq))
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
    trial_start = 0
    trial_samples = int(float(trial_len) * bci_freq)

    for ix, cue in enumerate(cues[1:], start=1):
        if cue != cues[ix - 1]:
            if cue == cue_map['HOVLEFT']:
                trials += [(left, ix)]
                eog_left += [eog[(eog_times >= cue_times[ix]) &
                                 (eog_times <= cue_times[ix] + trial_len)]]
            elif cue == cue_map['HOVRIGHT']:
                trials += [(right, ix)]
                eog_right += [eog[(eog_times >= cue_times[ix]) &
                                  (eog_times <= cue_times[ix] + trial_len)]]

    # convert to numpy matrices
    eog_left = convert_to_numpy("left eog", eog_left)
    eog_right = convert_to_numpy("right eog", eog_right)

    return trials, eog_left, eog_right

participants_id = ['P01','P02','P03','P04','P05']
trial_len = 3           # length of trials in seconds

ERD_reliabilities = []
ERS_reliabilities = []
EOG_reliabilities = []
EEG_Accuracies = []
EOG_Accuracies = []

for p in participants_id:
    # Load EEG data from xdf file
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

    # Select the laterality of the hand for which motor imagery / motor attemt was executed
    # LEFT  (hand) -> brain signal from right hemisphere (C4)
    # RIGHT (hand) -> brain signal from right hemisphere (C3)
    Laterality = Side(get_parameters_from_xdf_stream(feedback_stream)['laterality'])

    # Convert the preprocessed EEG data into MNE raw_data object and use the markers to define experimental annotations
    EEG_preprocessed_data, ev, ev_id = stream2raw(preprocessed_stream,marker_stream=marker_stream, marker_out=3)
    bci_freq = preprocessed_stream['info']['effective_srate']
    # Extract target EEG channels C3 and C4
    targets = {'prep': ['µC3', 'µC4']}
    prep_channel = find_channel_index(preprocessed_stream, targets['prep'])
    data_prep = EEG_preprocessed_data.pick_channels([ch for ch in targets['prep']])

    # Continue with the EEG controlateral to the paretic hand only
    if Laterality is Side.RIGHT:
        data_prepC3 = data_prep.copy().pick(targets['prep'][0])
        # Calculate reference value (RV) on preprocessed data from Start on and normalize the data based on the RV
        start = int(round(15 * bci_freq))  # Skip the first 15 seconds
        rv = data_prepC3.get_data([0],start=start).mean()
        # Extract Close and Relax trials from the preprocessed data
        Close_processed_events = mne.events_from_annotations(data_prepC3, regexp="CLOSE")
        Relax_processed_events = mne.events_from_annotations(data_prepC3, regexp="RELAX")
        close_preprocessed_epochs = mne.Epochs(data_prepC3, Close_processed_events[0], tmin=1, tmax=5, baseline=None,
                                               preload=True)
        relax_preprocessed_epochs = mne.Epochs(data_prepC3, Relax_processed_events[0], tmin=1, tmax=5, baseline=None,
                                               preload=True)
        # Normalize close and relax trials
        norm_close = (close_preprocessed_epochs._data / rv)-1
        norm_relax = (relax_preprocessed_epochs._data/rv)-1

    elif Laterality is Side.LEFT:
        data_prepC4 = data_prep.copy().pick(targets['prep'][1])
        # Calculate reference value (RV) on preprocessed data from Start on and normalize the data based on the RV
        start = int(round(15 * bci_freq))  # Skip the first 15 seconds
        rv = data_prepC4.get_data([0],start=start).mean()
        # Extract Close and Relax trials from the preprocessed data
        Close_processed_events = mne.events_from_annotations(data_prepC4, regexp="CLOSE")
        Relax_processed_events = mne.events_from_annotations(data_prepC4, regexp="RELAX")
        close_preprocessed_epochs = mne.Epochs (data_prepC4,Close_processed_events[0], tmin=1,tmax=5, baseline=None, preload=True)
        relax_preprocessed_epochs = mne.Epochs (data_prepC4, Relax_processed_events[0], tmin=1, tmax=5, baseline=None, preload=True)
        # Normalize close and relax trials
        norm_close = (close_preprocessed_epochs._data/rv)-1
        norm_relax = (relax_preprocessed_epochs._data/rv)-1

    #Compute ERD Accuracy #Time below the threshold during motor task, and above the threshold at rest
    K = norm_close.shape[2]*norm_close.shape[0]
    EEG_threshold = np.sort(np.concatenate(norm_close.squeeze()))[-int(K*0.75)] #Compute the 75 percentile and set it as classification threshold

    True_Positive = 0
    False_Negative = 0
    for trial in range(len(norm_close)):
        count = 0
        for t in range(norm_close.shape[2]):
            if norm_close.squeeze()[trial,t] < EEG_threshold:
                count +=1

        if count >= 1:
            True_Positive +=1
        else:
            False_Negative +=1

    True_Negative = 0
    False_Positive = 0
    for trial in range(len(norm_relax)):
        count = 0
        for t in range(norm_relax.shape[2]):
            if norm_relax.squeeze()[trial, t] < EEG_threshold:
                count += 1
        if count >= 1:
            False_Positive +=1
        else:
            True_Negative +=1

    Control_Accuracy = (True_Positive + True_Negative)/(True_Positive+True_Negative+False_Negative+False_Positive)
    EEG_Accuracies.append(Control_Accuracy)

    #Load EOG data from xdf file
    EOG_filepath = str("/Users/AnnalisaColucci/Desktop/Handy Rehab B:NE Data Analysis/Data/Group Analysis/EOG/" + p + "_EOG.xdf")
    # Load marker stream containing the experimental cues
    EOG_marker_stream, _ = pyxdf.load_xdf(EOG_filepath, dejitter_timestamps=True, select_streams=[{'name': 'TaskOutput'}])
    EOG_marker_stream = EOG_marker_stream[0]

    # Load preprocessed stream containing the data after online preprocessing
    EOG_preprocessed_stream, _ = pyxdf.load_xdf(EOG_filepath, dejitter_timestamps=True, select_streams=[{'name': 'PreprocessedData'}])
    EOG_preprocessed_stream = EOG_preprocessed_stream[0]

    # Convert the preprocessed EOG data into MNE raw_data object and use the markers to define experimental annotations
    EOG_preprocessed_data, ev, ev_id = stream2raw(EOG_preprocessed_stream,marker_stream=EOG_marker_stream, marker_out=0)
    #EOG_preprocessed_data.resample(sfreq=res_frequency)

    # get mappings of the HOV cues
    cue_map = dict(zip(
        EOG_marker_stream['info']['desc'][0]['mappings'][0]['cues'][0].keys(),
        map(lambda x: int(x[0]),
            EOG_marker_stream['info']['desc'][0]['mappings'][0]['cues'][0].values())
    ))

    # extract time series and time stamps of EOG preprocessed data and of cue markers
    eog = EOG_preprocessed_stream['time_series'][:, 0].squeeze()
    eog_times = EOG_preprocessed_stream['time_stamps']
    cues = np.array(EOG_marker_stream['time_series'])[:, 0].squeeze().astype(int)
    cue_times = EOG_marker_stream['time_stamps']

    # start timestamps with 0
    start_time = min(cue_times[0], eog_times[0])
    eog_times -= start_time
    cue_times -= start_time

    # extract update frequency
    BCI_freq = round(EOG_preprocessed_stream['info']['effective_srate'])

    # define cue sides ad extract HOV epoched data
    left = int(EOG_marker_stream['info']['desc'][0]['mappings'][0]['cues'][0]['HOVLEFT'][0])
    right = int(EOG_marker_stream['info']['desc'][0]['mappings'][0]['cues'][0]['HOVRIGHT'][0])
    (trials, eog_left, eog_right) = extract_hov_trials()

    # Etract the data mean and normalize the data
    if Laterality is Side.RIGHT:
        eog_mean = eog_left.copy().mean(axis=(0,1))
        eog_norm = (eog_left/eog_mean)-1

    elif Laterality is Side.LEFT:
        eog_mean = eog_right.copy().mean(axis=(0,1))
        eog_norm = (eog_right/eog_mean)-1

    #Compute EOG Accuracy
    K = eog_norm.shape[0]*eog_norm.shape[1]
    EOG_threshold = np.sort(np.concatenate(eog_norm.squeeze()))[int(K*0.75)] #Compute the 75 percentile and set it as classification threshold

    EOG = np.zeros(len(eog_norm))
    EOG_truePositive = 0
    EOG_falseNegative = 0
    for trial in range(len(eog_norm)):
        count = 0
        for t in range(eog_norm.shape[1]):
            if eog_norm[trial, t] > EOG_threshold:
                count += 1
        if count >= 1:
            EOG_truePositive += 1
        elif count <=1:
            EOG_falseNegative += 1
    EOG_Accuracy = EOG_truePositive/(EOG_truePositive+EOG_falseNegative)
    EOG_Accuracies.append(EOG_Accuracy)

# Transform ERD Accuracy in %
EEG_ControlAccuracy = np.array(EEG_Accuracies)*100
EOG_ControlAccuracy = np.array(EOG_Accuracies)*100
# Print accuracy results mean and standard deviation
print("ERD Accuracy mean "+ str(np.array(EEG_ControlAccuracy).mean()))
print("ERD std: "+ str(np.std(np.array(EEG_ControlAccuracy))))
print("EOG mean "+ str(np.array(EOG_ControlAccuracy).mean()))
print("EOG std: "+ str(np.std(np.array(EOG_ControlAccuracy))))

# Plot Control Accuracy
plt.figure(figsize=[7,6])
sns.set(style='whitegrid')
palette =['green','red']
sns.boxplot([EEG_ControlAccuracy,EOG_ControlAccuracy],palette=palette,width=0.8, linewidth=0.5)
sns.stripplot([EEG_ControlAccuracy,EOG_ControlAccuracy],palette=palette, edgecolor='black', linewidth=1)
plt.xticks([0,1],['EEG', 'EOG'],fontsize=25)
plt.ylim([50,103])
plt.yticks(fontsize=20)
plt.axhline(60,linestyle='-.',color='grey', linewidth=0.7)
plt.ylabel("Accuracy (% of trials)", fontsize=25)
plt.title("Control Accuracy", fontsize=27)
plt.grid(False)
#plt.savefig("ControlAccuracy_Corrected.svg", format="svg")
plt.show()
print("END")