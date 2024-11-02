# This is a program to do pre-processing for the data

import math
import numpy as np
from scipy import signal
from scipy.signal import welch
from scipy.integrate import simpson as simps
import scipy.sparse as sp


def band_pass(data, low_frequency, high_frequency, sampling_rate, filter_order=8):
    # data: 2d array, channel x datapoint
    # filtered_data: 2d array, channel x datapoint
    wn1 = 2 * low_frequency / sampling_rate
    wn2 = 2 * high_frequency / sampling_rate
    b, a = signal.butter(filter_order, [wn1, wn2], 'bandpass')
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data


def band_pass_cheby2_sos(data, bandFiltCutF=[0.3, 40], fs=128, filtAllowance=[0.2, 5], axis=1):
    # data: channel x time
    aStop = 30  # stopband attenuation
    aPass = 3  # passband attenuation
    nFreq = fs / 2  # Nyquist frequency
    fPass = (np.array(bandFiltCutF) / nFreq).tolist()
    fStop = [(bandFiltCutF[0] - filtAllowance[0]) / nFreq, (bandFiltCutF[1] + filtAllowance[1]) / nFreq]
    # find the order
    [N, ws] = signal.cheb2ord(fPass, fStop, aPass, aStop)
    sos = signal.cheby2(N, aStop, fStop, 'bandpass', output='sos')
    dataOut = signal.sosfilt(sos, data, axis=axis)
    return dataOut


def get_DE(data, axis=0):
    DE = 0.5 * np.log(2 * 3.14 * 2.718 * np.var(data, axis=axis))
    return DE


def log_power(data, axis=0, relative=False):
    power = np.mean(np.power(data, 2), axis=axis)
    if relative:
        power = power / np.sum(power, axis=-1, keepdims=True)
    else:
        power = np.log(power)
    return power


def filter_bank(data, sampling_rate=256, filter_order=5,
                index_fbank=[[5, 8], [9, 12], [13, 16], [17, 20], [21, 24],
                             [25, 28], [29, 32], [33, 36], [37, 40]]):
    # filter band covers 4-40Hz signal, with 4 Hz each bank
    # data: channel x datapoint
    # return: frequency x channel x datapoint
    data_fbank = []
    for i in range(len(index_fbank)):
        data_fbank.append(band_pass(data, index_fbank[i][0], index_fbank[i][1], sampling_rate, filter_order))
    return np.stack(data_fbank, axis=0)


def filter_bank_cheby2(data, sampling_rate=256, filter_order=5,
                index_fbank=[[5, 8], [9, 12], [13, 16], [17, 20], [21, 24],
                             [25, 28], [29, 32], [33, 36], [37, 40]]):
    # filter band covers 4-40Hz signal, with 4 Hz each bank
    # data: channel x datapoint
    # return: frequency x channel x datapoint
    data_fbank = []
    filt_allowance = [[0.5, 2], [2, 2]]
    for i in range(len(index_fbank)):
        if i == 0:
            f_allow = filt_allowance[0]
        else:
            f_allow = filt_allowance[1]
        data_fbank.append(band_pass_cheby2_sos(
            data=data,
            bandFiltCutF=[index_fbank[i][0], index_fbank[i][1]],
            fs=sampling_rate,
            filtAllowance=f_allow
        ))
    return np.stack(data_fbank, axis=0)


def get_feature(data, frequency_idx, feature, sampling_rate):
    # data: channel x data
    # return : channel x feature
    filter_order = 5
    frequencies = len(frequency_idx)
    channels = data.shape[0]
    data_time = data.shape[1]

    if feature == 'multi-view':
        data_temp = np.zeros((frequencies, channels, data_time))
    else:
        data_temp = np.zeros((channels, frequencies))

    if feature == 'DE':
        data_ = filter_bank(data, sampling_rate, filter_order, frequency_idx)
        for f in range(frequencies):
            for chan in range(channels):
                data_temp[chan, f] = DE(data_[f, chan, :])
    elif feature == 'DE-C':
        data_ = filter_bank_cheby2(data, sampling_rate, filter_order, frequency_idx)
        for f in range(frequencies):
            for chan in range(channels):
                data_temp[chan, f] = DE(data_[f, chan, :])

    elif feature == 'PSD':
        data_temp = bandpower(data, sampling_rate, frequency_idx, 1, False)
    elif feature == 'rPSD':
        data_temp = bandpower(data, sampling_rate, frequency_idx, 1, True)

    elif feature == 'multi-view':
        for f in range(frequencies):
            data_temp[f] = band_pass(data, frequency_idx[f][0], frequency_idx[f][1], sampling_rate, filter_order)
    elif feature == 'Hjorth':
        data_temp = TemporalFeature(data)
    elif feature == 'sta':
        for f in range(frequencies):
            data_ = filter_bank(data, sampling_rate, filter_order, frequency_idx)
        data_temp_ = []
        data_temp_.append(np.mean(data_, axis=-1))
        data_temp_.append(np.std(data_, axis=-1))
        data_temp_.append(np.max(data_, axis=-1))
        data_temp_.append(np.min(data_, axis=-1))
        data_temp = np.concatenate(data_temp_, axis=-1)
    elif feature == 'mean':
        for f in range(frequencies):
            data_ = filter_bank(data, sampling_rate, filter_order, frequency_idx)
        data_temp = np.mean(data_, axis=-1)
    elif feature == 'std':
        for f in range(frequencies):
            data_ = filter_bank(data, sampling_rate, filter_order, frequency_idx)
        data_temp = np.std(data_, axis=-1)
    elif feature == 'max':
        for f in range(frequencies):
            data_ = filter_bank(data, sampling_rate, filter_order, frequency_idx)
        data_temp = np.max(data_, axis=-1)
    elif feature == 'min':
        for f in range(frequencies):
            data_ = filter_bank(data, sampling_rate, filter_order, frequency_idx)
        data_temp = np.min(data_, axis=-1)
    return data_temp


def DE(data):
    # data: 1d np.array
    # DE: float scaler
    DE = 0.5 * math.log(2 * 3.14 * 2.718 * np.var(data))
    return DE


def TemporalFeature(data):
    #data: channel x time
    #return feature: channel x 3
    feature = []
    for chan in data:
        diff = np.diff(chan)
        ddiff = np.diff(diff)
        var = np.var(chan)
        dvar = np.var(diff)
        ddvar = np.var(ddiff)

        activity = var
        mobility = np.sqrt(dvar/var)
        complexity = np.sqrt(ddvar/dvar)/mobility
        feature.append([activity, mobility, complexity])
    feature = np.stack(feature, axis=0)
    return feature


def bandpower(data, fs, band_sequence, window_sec=1, relative=False):
    # Compute the modified periodogram (Welch)
    # data: (chan, time)
    # return: (chan * num_f)

    # Define window length

    nperseg = window_sec * fs

    freqs, psd = welch(data, fs, nperseg=nperseg)

    freq_res = freqs[1] - freqs[0]

    band_powers = []

    for band in band_sequence:
        low, high = band
        idx_band = np.logical_and(freqs >= low, freqs <= high)

        # Integral approximation of the spectrum using Simpson's rule.
        band_power = simps(psd[:, idx_band], dx=freq_res)

        if relative:
            band_power /= simps(psd, dx=freq_res)

        band_powers.append(band_power)   #(6, 32)

    band_powers = np.asarray(band_powers)  #(6, 32)
    band_powers = band_powers.T   #(32, 6)
    return band_powers






