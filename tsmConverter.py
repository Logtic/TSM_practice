import numpy as np
from tsmModule import basicTSM

EPSILON = 0.0001

def find_peaks(amplitude):
    padded = np.concatenate((-np.ones(2), amplitude, -np.ones(2)))
    shifted_l2 = padded[:-4]
    shifted_l1 = padded[1:-3]
    shifted_r1 = padded[3:-1]
    shifted_r2 = padded[4:]
    peaks = ((amplitude >= shifted_l2) & (amplitude >= shifted_l1) &
             (amplitude >= shifted_r1) & (amplitude >= shifted_r2))
    return peaks

def get_closest_peaks(peaks):
    closest_peak = np.empty_like(peaks, dtype=int)
    previous = -1
    for i, is_peak in enumerate(peaks):
        if is_peak:
            if previous >= 0:
                closest_peak[previous:(previous+i) // 2 + 1] = previous
                closest_peak[(previous + i) // 2 + 1:i] = i
            else:
                closest_peak[:i] = i
            previous = i
    closest_peak[previous:] = previous
    return closest_peak

class TSMConverter():
    def __init__(self, channels, frame_length, analysis_hop, synthesis_hop):
        self._channels = channels
        self._frame_length = frame_length
        self._synthesis_hop = synthesis_hop
        self._analysis_hop = analysis_hop
        self._find_peaks = find_peaks

        self._center_frequency = np.fft.rfftfreq(self._frame_length) * 2 * np.pi
        fft_length = len(self._center_frequency)

        self._first = True

        self._previous_phase = np.empty((channels, fft_length))
        self._output_phase = np.empty((channels, fft_length))

        self._buffer = np.empty(fft_length)

    def clear(self):
        self._first = True

    def convert_harmonic_frame(self, frame):
        for k in range(0, self._channels):
            stft = np.fft.rfft(frame[k])
            amplitude = np.abs(stft)
            phase = np.angle(stft)
            del stft

            peaks = self._find_peaks(amplitude)
            closest_peak = get_closest_peaks(peaks)
            if self._first:
                self._output_phase[k, :] = phase
            else:
                self._buffer[peaks] = (
                    phase[peaks] - self._previous_phase[k, peaks] -
                    self._analysis_hop * self._center_frequency[peaks]
                )

                self._buffer[peaks] += np.pi
                self._buffer[peaks] %= 2*np.pi
                self._buffer[peaks] -= np.pi

                self._buffer[peaks] /= self._analysis_hop
                self._buffer[peaks] += self._center_frequency[peaks]

                self._buffer[peaks] *= self._synthesis_hop
                self._output_phase[k][peaks] += self._buffer[peaks]

                self._output_phase[k] = (
                    self._output_phase[k][closest_peak] +
                    phase - phase[closest_peak]
                )

                output_stft = amplitude * np.exp(1j * self._output_phase[k])
                frame[k, :] = np.fft.irfft(output_stft).real

            self._previous_phase[k, :] = phase
            del phase
            del amplitude

        self._first = False

        return frame

    def convert_percussive_frame(self, frame):
        for k in range(0, self._channels):
            stft = np.fft.rfft(frame[k])
            amplitude = np.abs(stft)
            phase = np.angle(stft)
            del stft

            peaks = self._find_peaks(amplitude)
            closest_peak = get_closest_peaks(peaks)
            if self._first:
                self._output_phase[k, :] = phase
            else:
                self._buffer[peaks] = (
                    phase[peaks] - self._previous_phase[k, peaks] -
                    self._analysis_hop * self._center_frequency[peaks]
                )

                self._buffer[peaks] += np.pi
                self._buffer[peaks] %= 2*np.pi
                self._buffer[peaks] -= np.pi

                self._buffer[peaks] /= self._analysis_hop
                self._buffer[peaks] += self._center_frequency[peaks]

                self._buffer[peaks] *= self._synthesis_hop
                self._output_phase[k][peaks] += self._buffer[peaks]

                self._output_phase[k] = (
                    self._output_phase[k][closest_peak] +
                    phase - phase[closest_peak]
                )

                output_stft = amplitude * np.exp(1j * self._output_phase[k])
                frame[k, :] = np.fft.irfft(output_stft).real

            self._previous_phase[k, :] = phase
            del phase
            del amplitude

        self._first = False

        return frame

    def set_analysis_hop(self, analysis_hop):
        self._analysis_hop = analysis_hop

def harmonic(channels, origin_time, convert_time, frame_length, sample_rate):
    analysis_hop = frame_length // 4
    harmonic_hop = int(convert_time * analysis_hop * sample_rate / (origin_time * sample_rate - frame_length))

    harmonic_converter = TSMConverter(channels, frame_length, analysis_hop, harmonic_hop)

    return basicTSM(harmonic_converter,
                    channels, frame_length, analysis_hop, harmonic_hop, 0, 0)

def percussive(channels, origin_time, convert_time, frame_length, sample_rate):
    analysis_hop = frame_length // 4
    harmonic_hop = int(convert_time * analysis_hop * sample_rate / (origin_time * sample_rate - frame_length))

    harmonic_converter = TSMConverter(channels, frame_length, analysis_hop, harmonic_hop)

    return basicTSM(harmonic_converter,
                    channels, frame_length, analysis_hop, harmonic_hop, 0, 0)
