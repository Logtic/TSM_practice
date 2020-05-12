import numpy as np
import os
import librosa
from librosa import core, decompose, spectrum, output, onset
from wav import WavWriter, WavReader
from tsmConverter import harmonic, percussive
import pyworld as pw
import argparse
import soundfile as sf
import matplotlib.pyplot as plt

def calculateF0(f0):
    i=0
    count = 0
    for x in f0:
        if x>0:
            count += x
            i += 1
    return count // i

def padding(D_harmonic, D_percu):
    pad_x = max(D_harmonic.shape[0], D_percu.shape[0])
    temp_harmonic = np.zeros((pad_x))
    temp_percu = np.zeros((pad_x))
    temp_harmonic[:D_harmonic.shape[0]] = D_harmonic
    temp_percu[:D_percu.shape[0]] = D_percu
    return temp_harmonic, temp_percu

def convert_lower_threshold(y, input_filename, output_filename, sr, parameters):
    harmonic_path = input_filename.replace('.wav', 'harmonic_.wav')
    out_harmonic_path = input_filename.replace('.wav', 'out_harmonic_.wav')
    percussive_path = input_filename.replace('.wav', 'percussive_.wav')
    out_percussive_path = input_filename.replace('.wav', 'out_percussive_.wav')

    D = spectrum.stft(y)
    D_h, D_p = decompose.hpss(D, power=2.0)

    D_harmonic, D_percussive = padding(spectrum.istft(D_h), spectrum.istft(D_p))
    sf.write(harmonic_path, D_harmonic, sr)
    sf.write(percussive_path, D_percussive, sr)
    with WavReader(percussive_path) as reader:
        with WavWriter(out_percussive_path, reader.channels, reader.samplerate) as writer:
            tsm = harmonic(reader.channels, **parameters)
            tsm.run_harmonic(reader, writer)

    with WavReader(harmonic_path) as reader:
        with WavWriter(out_harmonic_path, reader.channels, reader.samplerate) as writer:
            tsm = harmonic(reader.channels, **parameters)
            tsm.run_harmonic(reader, writer)

    y1, sr1 = core.load(out_harmonic_path, sr=sr)
    y2, sr2 = core.load(out_percussive_path, sr=sr)
    D1, D2 = spectrum.stft(y1), spectrum.stft(y2)
    D_h, D_p = padding(spectrum.istft(D1), spectrum.istft(D2))

    sf.write(output_filename, (D_h + D_p), sr)

    '''if os.path.isfile(harmonic_path):
        os.remove(harmonic_path)
    if os.path.isfile(percussive_path):
        os.remove((percussive_path))'''
    if os.path.isfile(out_harmonic_path):
        os.remove(out_harmonic_path)
    if os.path.isfile(out_percussive_path):
        os.remove(out_percussive_path)

def convert_upper_threshold(input_filename, output_filename, parameters):
    with WavReader(input_filename) as reader:
        with WavWriter(output_filename, reader.channels, reader.samplerate) as writer:
            tsm = harmonic(reader.channels, **parameters)
            tsm.run_harmonic(reader, writer)

def main():
    # man 900 middle 750 default 500 frame_length = 900
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--speed', type=float, default=1.)
    parser.add_argument('-t', '--time', type=float, default=-10.)
    parser.add_argument('-o', '--output', type=str, default='output.wav')
    parser.add_argument('-i', '--input', type=str, default='input.wav')

    args = parser.parse_args()

    parameters = {}

    input_filename = args.input
    output_filename = args.output

    if not os.path.isfile(input_filename):
        raise RuntimeError('no input file')




    x, fs = core.load(input_filename)
    #f0, sp, ap = pw.wav2world(x, fs)
    frame_length = 1500 #100000 // int(calculateF0(f0)) // 2 * 2
    y, sr = core.load(input_filename, sr=fs)


    onset_frames = onset.onset_detect(x, sr=sr, wait=1, pre_avg=1, post_avg=1, pre_max=1, post_max=1)
    onset_times = librosa.frames_to_time(onset_frames)

    plt.plot(y)
    for i in onset_times:
        plt.plot([i*22050, i*22050], [-1, 1], color="red")


    S = librosa.stft(x)
    logS = librosa.amplitude_to_db(abs(S))

    plt.savefig('woman.png')

    if args.time < 0:
        parameters['origin_time'] = core.get_duration(y, sr)
        parameters['convert_time'] = parameters['origin_time'] / args.speed
    else:
        parameters['origin_time'] = core.get_duration(y, sr)
        parameters['convert_time'] = args.time
    parameters['sample_rate'] = sr
    parameters['frame_length'] = int(fs / 22050*frame_length)

    #if parameters['convert_time'] / parameters['origin_time'] > 0.8:
    convert_upper_threshold(input_filename, output_filename, parameters)

    #else:
    #convert_lower_threshold(y, input_filename, output_filename, sr, parameters)

if __name__ == "__main__":
    main()
