from scipy.signal import welch
from scipy.integrate import simps
import numpy as np
import mne
from scipy import signal
import math


class GenericEegController(object):

    def __init__(self, filename, config):
        self.filename = filename
        self.buffer_sec = config['buffer_sec']
        self.frequency = config['sampling_frequency']
        self.window_sec = config['window_sec']
        self.step = int(config['hop_sec'] * self.frequency)
        self.interest_bands = config['interest_bands']
        self.f_trans_interest_bands = config['f_trans_interest_bands']
        self.channel_slice = config['channel_slice']
        self.eeg_feature_list = config['features']
        self.filter_type = config['filter_type']
        self.filter_order = config['filter_order']
        self.extracted_data = self.preprocessing()

    def calculate_bandpower(self, data):
        # data: np.array (data, chan)
        data_np = data[:].T
        power_spectram_densities = []
        start = 0
        end = int(start + self.frequency * self.window_sec)
        while end < data_np.shape[1]:
            bandpower = bandpower_multiple(data_np[:, start:end], sampling_frequence=self.frequency,
                                           band_sequence=self.interest_bands, relative=True)
            power_spectram_densities.append(bandpower)
            start = start + self.step
            end = int(start + self.frequency * self.window_sec)
        power_spectram_densities = np.stack(power_spectram_densities)
        return power_spectram_densities

    def calculate_eeg(self, data):
        # data: np.array (data, chan)
        data_np = data[:].T
        power_spectram_densities = []
        start = 0
        end = int(start + self.frequency * self.window_sec)
        while end < data_np.shape[1]:
            eeg_seg = data_np[:, start:end]   # chan x time
            eeg_seg = np.concatenate(eeg_seg)    # (chan*time)
            power_spectram_densities.append(eeg_seg)
            start = start + self.step
            end = int(start + self.frequency * self.window_sec)
        power_spectram_densities = np.stack(power_spectram_densities)
        return power_spectram_densities

    def calculate_DE(self, data):
        data_np = data[:].T
        data_filtered = filter_band(
            data=data_np, bands=self.interest_bands, stop_fre=self.f_trans_interest_bands,
            filter_type=self.filter_type, fs=self.frequency, order=self.filter_order
        )   # (f, chan, time)
        DEs = []
        start = 0
        end = int(start + self.frequency * self.window_sec)
        while end < data_np.shape[1]:
            des = []
            for f in data_filtered:
                de = DE(f[:, start:end])   # (chan,)
                des.append(de)
            des = np.stack(des).T   #(chan, f)
            DEs.append(np.reshape(des, (des.shape[0]*des.shape[1])))
            start = start + self.step
            end = int(start + self.frequency * self.window_sec)
        DEs = np.stack(DEs)
        return DEs

    def calculate_fHjorth_per_fre(self, data):
        data_np = data[:].T
        data_filtered = filter_band(
            data=data_np, bands=self.interest_bands,
            filter_type=self.filter_type, fs=self.frequency, order=self.filter_order
        )   # (f, chan, time)
        Hjorths = []
        start = 0
        end = int(start + self.frequency * self.window_sec)
        while end < data_np.shape[1]:
            hjorth = []
            for f in data_filtered:
                de = TemporalFeature(f[:, start:end])   # (chan,)
                hjorth.append(de)
            hjorth = np.transpose(np.stack(hjorth), (1, 0, 2))  #(chan, f, 3)
            Hjorths.append(np.reshape(hjorth, (hjorth.shape[0]*hjorth.shape[1]*hjorth.shape[2])))
            start = start + self.step
            end = int(start + self.frequency * self.window_sec)
        Hjorths = np.stack(Hjorths)
        Hjorths = (Hjorths - np.mean(Hjorths, axis=-1, keepdims=True)) / np.std(Hjorths, axis=-1, keepdims=True)
        return Hjorths


    def calculate_fHjorth(self, data):
        data_np = data[:].T  # chan, time
        Hjorths = []
        start = 0
        end = int(start + self.frequency * self.window_sec)
        while end < data_np.shape[1]:
            tmf = TemporalFeature(data_np[:, start:end])   # (chan, 3)
            Hjorths.append(np.reshape(tmf, (tmf.shape[0]*tmf.shape[1])))   #(chan*3)
            start = start + self.step
            end = int(start + self.frequency * self.window_sec)
        Hjorths = np.stack(Hjorths)
        return Hjorths

    def calculate_RP(self, data):
        data_np = data[:].T
        data_filtered = filter_band(
            data=data_np, bands=self.interest_bands,
            filter_type=self.filter_type, fs=self.frequency, order=self.filter_order
        )   # (f, chan, time)
        RPs = []
        start = 0
        end = int(start + self.frequency * self.window_sec)
        while end < data_np.shape[1]:
            rp_t = RP(data=data_filtered[:, :, start:end])
            rp_t = np.reshape(rp_t, (rp_t.shape[0]*rp_t.shape[1]))
            RPs.append(rp_t)
            start = start + self.step
            end = int(start + self.frequency * self.window_sec)
        RPs = np.stack(RPs)
        RPs = RPs / np.sum(RPs, axis=-1, keepdims=True)
        return RPs


    def preprocessing(self):
        raw_data = self.read_data()
        channel_type_dictionary = self.get_channel_type_dictionary(raw_data)
        raw_data = self.set_channel_types_from_dictionary(raw_data, channel_type_dictionary)
        crop_range = self.get_crop_range_in_second(raw_data)
        cropped_raw_data = self.crop_data(raw_data, crop_range)
        cropped_eeg_raw_data = self.get_eeg_data(cropped_raw_data)
        filtered_data_np = self.filter_eeg_data(cropped_eeg_raw_data)
        average_referenced_data_np = self.average_reference(filtered_data_np)


        extracted_data = {}
        if "eeg_raw" in self.eeg_feature_list:
            eeg_segments = self.calculate_eeg(average_referenced_data_np)
            extracted_data.update({'eeg_raw': eeg_segments})

        if "eeg_bandpower" in self.eeg_feature_list:
            bandpower = self.calculate_bandpower(average_referenced_data_np)
            extracted_data.update({'eeg_bandpower': bandpower})

        if "eeg_DE" in self.eeg_feature_list:
            bandpower = self.calculate_DE(average_referenced_data_np)
            extracted_data.update({'eeg_DE': bandpower})

        if "eeg_RP" in self.eeg_feature_list:
            bandpower = self.calculate_RP(average_referenced_data_np)
            extracted_data.update({'eeg_RP': bandpower})

        if "eeg_Hjorth" in self.eeg_feature_list:
            bandpower = self.calculate_fHjorth(average_referenced_data_np)
            extracted_data.update({'eeg_Hjorth': bandpower})

        return extracted_data

    def produce_bandpower_array(self, band_power):
        # 32 channel
        num_bp_per_sample = len(self.interest_bands) * 32
        bp_array = np.reshape(band_power, (-1, num_bp_per_sample))
        return bp_array

    @staticmethod
    def set_interest_bands():
        interest_bands = [(0.3, 4), (4, 8), (8, 12), (12, 18), (18, 30), (30, 45)]
        return interest_bands

    @staticmethod
    def filter_eeg_data(data):
        r"""we'l
        Filter the eeg signal using lowpass and highpass filter.
        :return: (mne object), the filtered eeg signal.
        """
        filtered_eeg_data = data.copy().load_data().filter(l_freq=0.3, h_freq=45, method='iir')
        return filtered_eeg_data

    @staticmethod
    def average_reference(data):
        average_referenced_data = data.copy().load_data().set_eeg_reference()
        return average_referenced_data[:][0].T

    def read_data(self):
        r"""
        Load the bdf data using mne API.
        :return: (mne object), the raw signal containing different channels.
        """

        raw_data = mne.io.read_raw_bdf(self.filename)

        return raw_data

    def get_channel_slice(self):
        r"""
        Assign a tag to each channel according to the dataset paradigm.
        :return:
        """
        channel_slice = {'eeg': slice(0, 32), 'ecg': slice(32, 35), 'misc': slice(35, -1)}
        return channel_slice

    def get_channel_type_dictionary(self, data):
        r"""
        Generate a dictionary where the key is the channel names, and the value
            is the modality name (such as eeg, ecg, eog, etc...)
        :return: (dict), the dictionary of channel names to modality name.
        """
        channel_type_dictionary = {}
        for modal, slicing in self.channel_slice.items():
            channel_type_dictionary.update({channel: modal
                                            for channel in data.ch_names[
                                                self.channel_slice[modal]]})

        return channel_type_dictionary

    @staticmethod
    def set_channel_types_from_dictionary(data, channel_type_dictionary):
        r"""
        Set the channel types of the raw data according to a dictionary. I did this
            in order to call the automatic EOG, ECG remover. But it currently failed. Need to check.
        :return:
        """
        data = data.set_channel_types(channel_type_dictionary)
        return data

    def get_crop_range_in_second(self, data):
        r"""
        Assign the stimulated time interval for cropping.
        :return: (list), the list containing the data time interval
        """
        crop_range = [[30. - self.window_sec / 2, data.times.max() - 30 + self.buffer_sec]]
        return crop_range

    @staticmethod
    def crop_data(data, crop_range):
        r"""
        Crop the signal so that only the stimulated parts are preserved.
        :return: (mne object), the cropped data.
        """
        cropped_data = []
        for index, (start, end) in enumerate(crop_range):

            if index == 0:
                cropped_data = data.copy().crop(tmin=start, tmax=end)
            else:
                cropped_data.append(data.copy().crop(tmin=start, tmax=end))

        return cropped_data

    @staticmethod
    def get_eeg_data(data):
        r"""
        Get only the eeg data from the raw data.
        :return: (mne object), the eeg signal.
        """
        eeg_data = data.copy().pick_types(eeg=True)
        return eeg_data


def bandpower_multiple(data, sampling_frequence, band_sequence, window_sec=1, relative=False):
    # Compute the modified periodogram (Welch)

    nperseg = window_sec * sampling_frequence

    freqs, psd = welch(data, sampling_frequence, nperseg=nperseg)

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
    #band_powers = np.reshape(band_powers, (band_powers.shape[0]*band_powers.shape[1]))
    return band_powers


def filter_band(data, bands, stop_fre, filter_type, fs, order):
    """
    This function create the band-passed eeg data
    data: (chan, time)
    bands: list of sub-bands
    stop_fre: list of stop frequency of the sub-band
    filter_type: str 'butter' or 'cheby2'
    fs: sampling rate
    order: int the order of the filter
    return: (f, chan, time)
    """
    filtered_data = []
    for idx, f in enumerate(bands):
        if filter_type=='cheby2':
            stop_f = stop_fre[idx]
            data_ = bandpassfilter_cheby2_sos(
                data=data, bandFiltCutF=[f[0], f[1]], fs=fs, filtAllowance=[stop_f[0], stop_f[1]], axis=1,
            )
        else:
            data_ = band_pass_butter(
                data=data, f_low=f[0], f_high=f[1], fs=fs, filter_order=order
            )
        filtered_data.append(data_)

    filtered_data = np.stack(filtered_data)  # (f, chan, time)
    return filtered_data


def bandpassfilter_cheby2_sos(data, bandFiltCutF=[0.3, 40], fs=256, filtAllowance=[0.2, 5], axis=2):
    '''
    Band-pass filter the EEG signal of one subject using cheby2 IIR filtering
    and implemented as a series of second-order filters with direct-form II transposed structure.
    Param:
        data: nparray, size [trials x channels x times], original EEG signal
        bandFiltCutF: list, len: 2, low and high cut off frequency (Hz),
                If any value is None then only one-side filtering is performed.
        fs: sampling frequency (Hz)
        filtAllowance: list, len: 2, transition bandwidth (Hz) of low-pass and high-pass f
        axis: the axis along which apply the filter.
    Returns:
        data_out: nparray, size [trials x channels x times], filtered EEG signal
    '''

    aStop = 40  # stopband attenuation
    aPass = 1  # passband attenuation
    nFreq = fs / 2  # Nyquist frequency

    if (bandFiltCutF[0] == 0 or bandFiltCutF[0] is None) and (bandFiltCutF[1] == None or bandFiltCutF[1] >= fs / 2.0):
        # no filter
        print("Not doing any filtering. Invalid cut-off specifications")
        return data

    elif bandFiltCutF[0] == 0 or bandFiltCutF[0] is None:
        # low-pass filter
        print("Using lowpass filter since low cut hz is 0 or None")
        fPass = bandFiltCutF[1] / nFreq
        fStop = (bandFiltCutF[1] + filtAllowance[1]) / nFreq
        # find the order
        [N, ws] = signal.cheb2ord(fPass, fStop, aPass, aStop)
        sos = signal.cheby2(N, aStop, fStop, 'lowpass', output='sos')

    elif (bandFiltCutF[1] is None) or (bandFiltCutF[1] == fs / 2.0):
        # high-pass filter
        print("Using highpass filter since high cut hz is None or nyquist freq")
        fPass = bandFiltCutF[0] / nFreq
        fStop = (bandFiltCutF[0] - filtAllowance[0]) / nFreq
        # find the order
        [N, ws] = signal.cheb2ord(fPass, fStop, aPass, aStop)
        sos = signal.cheby2(N, aStop, fStop, 'highpass', output='sos')

    else:
        # band-pass filter
        # print("Using bandpass filter")
        fPass = (np.array(bandFiltCutF) / nFreq).tolist()
        fStop = [(bandFiltCutF[0] - filtAllowance[0]) / nFreq, (bandFiltCutF[1] + filtAllowance[1]) / nFreq]
        # find the order
        [N, ws] = signal.cheb2ord(fPass, fStop, aPass, aStop)
        sos = signal.cheby2(N, aStop, fStop, 'bandpass', output='sos')

    dataOut = signal.sosfilt(sos, data, axis=axis)

    return dataOut


def band_pass_cheby2(data, f_low, f_high, fs, filter_order):
    # data: chan x data
    f_trans = 2
    f_pass = np.asarray([f_low, f_high])
    f_stop = np.asarray([f_low - f_trans, f_high + f_trans])
    gpass = 3
    gstop = 30
    f_nqs = fs / 2
    wp = f_pass / f_nqs
    ws = f_stop / f_nqs
    order = filter_order
    b, a = signal.cheby2(order, gstop, ws, btype='bandpass')
    data_filtered = signal.lfilter(b, a, data)
    return data_filtered


def band_pass_butter(data, f_low, f_high, fs, filter_order=8):
    # data: 2d array, channel x datapoint
    # filtered_data: 2d array, channel x datapoint
    wn1 = 2 * f_low / fs
    wn2 = 2 * f_high / fs
    b, a = signal.butter(filter_order, [wn1, wn2], 'bandpass')
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data


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


def DE(data):
    # data: (chan, time)
    # DE: (chan,)
    DE = 0.5 * np.log(2 * 3.14 * 2.718 * np.sqrt(np.std(data, axis=-1)))
    return DE


def RP(data):
    # data: 2d np.array(frequency, chan, data)
    # return: (chan, frequency,)
    data_temp = np.power(data, 2)
    data_temp = np.sum(data_temp, axis=-1)
    power_sum = np.sum(data_temp, axis=0, keepdims=True)
    data_temp = data_temp / power_sum
    return data_temp.T
