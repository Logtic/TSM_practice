import numpy as np

from utils import (windows, CBuffer, NormalizeBuffer)

EPSILON = 0.0001


class basicTSM():
    # pylint: disable=too-many-instance-attributes
    def __init__(self, converter, channels, frame_length, analysis_hop,
                 synthesis_hop,
                 delta_before=0, delta_after=0):
        # pylint: disable=too-many-arguments
        self._converter = converter

        self._channels = channels
        self._frame_length = frame_length
        self._analysis_hop = analysis_hop
        self._synthesis_hop = synthesis_hop

        self._analysis_window = windows.hanning(frame_length)
        self._synthesis_window = windows.hanning(frame_length)

        self._delta_before = delta_before
        self._delta_after = delta_after

        self._skip_input_samples = 0
        self._skip_output_samples = 0

        self._normalize_window = windows.product(self._analysis_window,
                                                 self._synthesis_window)

        if self._normalize_window is None:
            self._normalize_window = np.ones(self._frame_length)

        delta = self._delta_before + self._delta_after
        self._in_buffer = CBuffer(self._channels, self._frame_length + delta)
        self._analysis_frame = np.empty(
            (self._channels, self._frame_length + delta))
        self._out_buffer = CBuffer(self._channels, self._frame_length)
        self._normalize_buffer = NormalizeBuffer(self._frame_length)

        self.clear()

    def clear(self):
        # Clear the buffers
        self._in_buffer.remove(self._in_buffer.length)
        self._out_buffer.remove(self._out_buffer.length)
        self._out_buffer.right_pad(self._frame_length)
        self._normalize_buffer.remove(self._normalize_buffer.length)

        self._in_buffer.write(np.zeros(
            (self._channels, self._delta_before + self._frame_length // 2)))
        self._skip_output_samples = self._frame_length // 2

        # Clear the converter
        self._converter.clear()

    def flush_to(self, writer):
        if self._in_buffer.remaining_length == 0:
            raise RuntimeError("There is still data to process in the input "
                               "buffer, flush_to method should only be called "
                               "when write_to returns True.")

        n = self._out_buffer.write_to(writer)
        if self._out_buffer.ready == 0:
            # The output buffer is empty
            self.clear()
            return n, True

        return n, False

    def get_max_output_percussive_length(self, input_length):
        input_length -= self._skip_input_samples
        if input_length <= 0:
            return 0

        n_frames = input_length // self._analysis_hop + 1
        return n_frames * self._synthesis_hop

    def _process_harmonic_frame(self):
        self._in_buffer.peek(self._analysis_frame)
        self._in_buffer.remove(self._analysis_hop)

        windows.apply(self._analysis_frame, self._analysis_window)

        synthesis_frame = self._converter.convert_harmonic_frame(self._analysis_frame)

        windows.apply(synthesis_frame, self._synthesis_window)

        self._out_buffer.add(synthesis_frame)
        self._normalize_buffer.add(self._normalize_window)

        normalize = self._normalize_buffer.to_array(end=self._synthesis_hop)
        normalize[normalize < EPSILON] = 1
        self._out_buffer.divide(normalize)
        self._out_buffer.set_ready(self._synthesis_hop)
        self._normalize_buffer.remove(self._synthesis_hop)

    def _process_percussive_frame(self, append):
        self._in_buffer.peek(self._analysis_frame)
        self._in_buffer.remove(self._analysis_hop)

        synthesis_frame = self._converter.convert_percussive_frame(self._analysis_frame, append, windows)

        windows.apply(synthesis_frame, self._synthesis_window)

        self._out_buffer.add(synthesis_frame)
        self._normalize_buffer.add(self._normalize_window)

        normalize = self._normalize_buffer.to_array(end=self._synthesis_hop)
        normalize[normalize < EPSILON] = 1
        self._out_buffer.divide(normalize)
        self._out_buffer.set_ready(self._synthesis_hop)
        self._normalize_buffer.remove(self._synthesis_hop)

    def read_harmonic(self, reader):
        n = reader.skip(self._skip_input_samples)
        self._skip_input_samples -= n
        if self._skip_input_samples > 0:
            return n

        n += self._in_buffer.read_from(reader)

        if not (self._in_buffer.remaining_length == 0 and
                self._out_buffer.remaining_length >= self._synthesis_hop):
            return n

        self._process_harmonic_frame()

        skipped = self._out_buffer.remove(self._skip_output_samples)
        self._out_buffer.right_pad(skipped)
        self._skip_output_samples -= skipped

        self._skip_input_samples = self._analysis_hop - self._frame_length
        if self._skip_input_samples < 0:
            self._skip_input_samples = 0
        return n

    def read_percussive(self, reader, append):
        n = reader.skip(self._skip_input_samples)
        self._skip_input_samples -= n
        if self._skip_input_samples > 0:
            return n

        n += self._in_buffer.read_from(reader)

        if not (self._in_buffer.remaining_length == 0 and
                self._out_buffer.remaining_length >= self._synthesis_hop):
            return n

        self._process_percussive_frame()

        skipped = self._out_buffer.remove(self._skip_output_samples)
        self._out_buffer.right_pad(skipped)
        self._skip_output_samples -= skipped

        self._skip_input_samples = self._analysis_hop - self._frame_length
        if self._skip_input_samples < 0:
            self._skip_input_samples = 0
        return n

    def set_speed(self, speed):
        self._analysis_hop = int(self._synthesis_hop * speed)
        self._converter.set_analysis_hop(self._analysis_hop)

    def write_to(self, writer):
        n = self._out_buffer.write_to(writer)
        self._out_buffer.right_pad(n)

        if (self._in_buffer.remaining_length > 0 and
                self._out_buffer.ready == 0):
            # There is not enough data to process in the input buffer, and the
            # output buffer is empty
            return n, True

        return n, False

    def run_harmonic(self, reader, writer, flush=True):
        finished = False
        i = 0
        while not (finished and reader.empty):
            self.read_harmonic(reader)
            _, finished = self.write_to(writer)
        if flush:
            finished = False
            while not finished:
                _, finished = self.flush_to(writer)
            self.clear()

    def run_percussive(self, reader, writer, flush=True):
        finished = False
        count = 1
        i = 0
        while not (finished and reader.empty):
            count = (count + 1) % 64
            if count == 1:
                append = True
            else:
                append = False
            self.read_percussive(reader, append)
            _, finished = self.write_to(writer)
        if flush:
            finished = False
            while not finished:
                _, finished = self.flush_to(writer)
            self.clear()

