#!/usr/bin/env python3
import mne
import numpy as np


DATA_READERS = {
    "edf": mne.io.read_raw_edf,
}

def load_psg_sample(file_pair, data_reader, picks=None):
    raw_data = data_reader(
        file_pair[0], stim_channel="Event marker", infer_types=True, preload=True
    )

    annot_data = mne.read_annotations(file_pair[1])
    raw_data.set_annotations(annot_data, emit_warning=False)
    if picks:
        raw_data.pick(picks)

    return raw_data, annot_data

def load_psg_samples(data_files, data_reader, picks=None):
    data = []
    for file_pair in data_files:
        raw, annotations = load_psg_sample(file_pair, data_reader, picks=picks)
        data.append((raw, annotations))
    return data

def eeg_power_band(epochs):
    """EEG relative power band feature extraction.

    This function takes an ``mne.Epochs`` object and creates EEG features based
    on relative power in specific frequency bands that are compatible with
    scikit-learn.

    Parameters
    ----------
    epochs : Epochs
        The data.

    Returns
    -------
    X : numpy array of shape [n_samples, 5 * n_traces] -> [[eeg1_fb1, eeg2_fb1 ..., eeg1_fbn, eeg2_fbn, ...], ...]
        Transformed data.
    """
    # specific frequency bands
    FREQ_BANDS = {
        "delta": [0.5, 4.5],
        "theta": [4.5, 8.5],
        "alpha": [8.5, 11.5],
        "sigma": [11.5, 15.5],
        "beta": [15.5, 30],
    }

    spectrum = epochs.compute_psd(picks="eeg", fmin=0.5, fmax=30.0)
    psds, freqs = spectrum.get_data(return_freqs=True)
    # Normalize the PSDs
    psds /= np.sum(psds, axis=-1, keepdims=True)

    X = []
    for fmin, fmax in FREQ_BANDS.values():
        psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
        X.append(psds_band.reshape(len(psds), -1))

    return np.concatenate(X, axis=1)

def eegs_power_band(epoch_list):
    X = []
    for epoch in epoch_list:
        X.append(eeg_power_band(epoch))
    return np.concatenate(X)
