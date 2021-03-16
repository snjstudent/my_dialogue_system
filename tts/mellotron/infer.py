
from unidecode import unidecode
import re
import soundfile as sf
import pdb
import scipy.io.wavfile as siw
from text import cmudict, text_to_sequence
from data_utils import TextMelLoader, TextMelCollate
from layers import TacotronSTFT
from waveglow.denoiser import Denoiser
from model import Tacotron2, load_model
from hparams import create_hparams
import torch
import librosa
import pandas as pd
from scipy.io.wavfile import write
import scipy as sp
import numpy as np
from itertools import cycle
import matplotlib
import matplotlib.pyplot as plt
import IPython.display as ipd
from mellotron_utils import get_data_from_musicxml
import os
import sys
sys.path.append('waveglow/')


def panner(signal, angle):
    angle = np.radians(angle)
    left = np.sqrt(2)/2.0 * (np.cos(angle) - np.sin(angle)) * signal
    right = np.sqrt(2)/2.0 * (np.cos(angle) + np.sin(angle)) * signal
    return np.dstack((left, right))[0]


def plot_mel_f0_alignment(mel_source, mel_outputs_postnet, f0s, alignments, figsize=(16, 16)):
    fig, axes = plt.subplots(4, 1, figsize=figsize)
    axes = axes.flatten()
    axes[0].imshow(mel_source, aspect='auto',
                   origin='bottom', interpolation='none')
    axes[1].imshow(mel_outputs_postnet, aspect='auto',
                   origin='bottom', interpolation='none')
    axes[2].scatter(range(len(f0s)), f0s, alpha=0.5,
                    color='red', marker='.', s=1)
    axes[2].set_xlim(0, len(f0s))
    axes[3].imshow(alignments, aspect='auto',
                   origin='bottom', interpolation='none')
    axes[0].set_title("Source Mel")
    axes[1].set_title("Predicted Mel")
    axes[2].set_title("Source pitch contour")
    axes[3].set_title("Source rhythm")
    plt.tight_layout()


def plot_mel_f0_alignment(mel_source, mel_outputs_postnet, f0s, alignments, figsize=(16, 16)):
    fig, axes = plt.subplots(4, 1, figsize=figsize)
    axes = axes.flatten()
    axes[0].imshow(mel_source, aspect='auto',
                   origin='bottom', interpolation='none')
    axes[1].imshow(mel_outputs_postnet, aspect='auto',
                   origin='bottom', interpolation='none')
    axes[2].scatter(range(len(f0s)), f0s, alpha=0.5,
                    color='red', marker='.', s=1)
    axes[2].set_xlim(0, len(f0s))
    axes[3].imshow(alignments, aspect='auto',
                   origin='bottom', interpolation='none')
    axes[0].set_title("Source Mel")
    axes[1].set_title("Predicted Mel")
    axes[2].set_title("Source pitch contour")
    axes[3].set_title("Source rhythm")
    plt.tight_layout()


def load_mel(path):
    audio, sampling_rate = librosa.core.load(path, sr=hparams.sampling_rate)
    audio = torch.from_numpy(audio)
    if sampling_rate != hparams.sampling_rate:
        raise ValueError("{} SR doesn't match target {} SR".format(
            sampling_rate, stft.sampling_rate))
    audio_norm = audio.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    melspec = stft.mel_spectrogram(audio_norm)
    melspec = melspec.cuda()
    return melspec


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    # import pickle
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    # checkpoint_dict = pickle.load(open(checkpoint_path, 'rb'))
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


hparams = create_hparams()

stft = TacotronSTFT(hparams.filter_length, hparams.hop_length, hparams.win_length,
                    hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
                    hparams.mel_fmax)
# model = load_model(hparams).cuda().eval()
# learning_rate = hparams.learning_rate
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
#                              weight_decay=hparams.weight_decay)
# mellotron, _, _, _ = load_checkpoint(
#     "outdir/checkpoint_500", model, optimizer)
checkpoint_path = "models/mellotron_libritts.pt"
checkpoint_path = "outdir/checkpoint_35000"
mellotron = load_model(hparams).cuda().eval()
mellotron.load_state_dict(torch.load(checkpoint_path)['state_dict'])
mellotron = mellotron.cuda().eval()
waveglow_path = 'models/waveglow_256channels_universal_v4.pt'
waveglow = torch.load(waveglow_path)['model'].cuda().eval()
denoiser = Denoiser(waveglow).cuda().eval()

arpabet_dict = cmudict.CMUDict('data/cmu_dictionary')
audio_paths = 'filelists/tts_dataset/info_test.txt'
#audio_paths = 'data/examples_filelist.txt'
dataloader = TextMelLoader(audio_paths, hparams)
datacollate = TextMelCollate(1)

file_idx = 0
audio_path, text, sid = dataloader.audiopaths_and_text[file_idx]
print(text)


# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')


def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)


# get audio path, encoded text, pitch contour and mel for gst
text_encoded = torch.LongTensor(text_to_sequence(
    "kyouhaiitenkidesune,kennkyuusitakunai.", hparams.text_cleaners, arpabet_dict))[None, :].cuda()


# load source data to obtain rhythm using tacotron 2 as a forced aligner
x, y = mellotron.parse_batch(datacollate([dataloader[file_idx]]))
# x = list(x)
# x[0] = text_encoded
# x[5] = torch.IntTensor(1).cuda()
# x[1] = torch.IntTensor([int(len(x[0][0]))]).cuda()
# x = tuple(x)
with torch.no_grad():
    # get rhythm (alignment map) using tacotron 2
    mel_outputs, mel_outputs_postnet, gate_outputs, rhythm = mellotron.inference(
        tuple([x[0], x[2], x[-2], x[-1]]))
    audio = denoiser(waveglow.infer(
        mel_outputs_postnet, sigma=0.8), 0.01)[0, 0]
    audio = audio.cpu().numpy()
    write('wavfile/test.wav', hparams.sampling_rate, audio)
    import sys
    sys.exit()
# pdb.set_trace()
# data = np.array(audio[0].cpu().numpy()*32768, dtype=np.int16)
# siw.write("wavfile/test.wav", 44100, np.array(np.c_[data, data]))
# ipd.Audio(audio[0].data.cpu().numpy(), rate=hparams.sampling_rate)
# pdb.set_trace()

text_encoded = torch.LongTensor(text_to_sequence(
    text, hparams.text_cleaners, arpabet_dict))[None, :].cuda()
pitch_contour = dataloader[file_idx][3][None].cuda()
mel = load_mel(audio_path)
speaker_ids = TextMelLoader(
    "filelists/libritts_train_clean_100_audiopath_text_sid_shorterthan10s_atleast5min_train_filelist.txt", hparams).speaker_ids
speakers = pd.read_csv('filelists/libritts_speakerinfo.txt', engine='python', header=None, comment=';', sep=' *\| *',
                       names=['ID', 'SEX', 'SUBSET', 'MINUTES', 'NAME'])
speakers['MELLOTRON_ID'] = speakers['ID'].apply(
    lambda x: speaker_ids[x] if x in speaker_ids else -1)
female_speakers = cycle(
    speakers.query("SEX == 'F' and MINUTES > 20 and MELLOTRON_ID >= 0")['MELLOTRON_ID'].sample(frac=1).tolist())
male_speakers = cycle(
    speakers.query("SEX == 'M' and MINUTES > 20 and MELLOTRON_ID >= 0")['MELLOTRON_ID'].sample(frac=1).tolist())
data = get_data_from_musicxml(
    'data/haendel_hallelujah.musicxml', 132, convert_stress=True)
panning = {'Soprano': [-60, -30], 'Alto': [-40, -10],
           'Tenor': [30, 60], 'Bass': [10, 40]}
n_speakers_per_part = 4
frequency_scaling = 0.4
n_seconds = 90
audio_stereo = np.zeros((hparams.sampling_rate*n_seconds, 2), dtype=np.float32)
for i, (part, v) in enumerate(data.items()):
    rhythm = data[part]['rhythm'].cuda()
    pitch_contour = data[part]['pitch_contour'].cuda()
    text_encoded = data[part]['text_encoded'].cuda()

    for k in range(n_speakers_per_part):
        pan = np.random.randint(panning[part][0], panning[part][1])
        if any(x in part.lower() for x in ('soprano', 'alto', 'female')):
            speaker_id = torch.LongTensor([0]).cuda()
        else:
            speaker_id = torch.LongTensor([0]).cuda()
        print("{} MellotronID {} pan {}".format(part, speaker_id.item(), pan))

        with torch.no_grad():
            mel_outputs, mel_outputs_postnet, gate_outputs, alignments_transfer = mellotron.inference_noattention(
                (text_encoded, mel, speaker_id, pitch_contour*frequency_scaling, rhythm))

            audio = denoiser(waveglow.infer(
                mel_outputs_postnet, sigma=0.8), 0.01)[0, 0]
            audio = audio.cpu().numpy()
            # pdb.set_trace()
            audio = panner(audio, pan)
            audio_stereo[:audio.shape[0]] += audio
            # pdb.set_trace()
            write("{} {}.wav".format(part, speaker_id.item()),
                  hparams.sampling_rate, audio)

with torch.no_grad():
    panning = {'Soprano': [-60, -30], 'Alto': [-40, -10],
               'Tenor': [30, 60], 'Bass': [10, 40]}
    pan = np.random.randint(panning['Soprano'][0], panning['Soprano'][1])
    audio = denoiser(waveglow.infer(
        mel_outputs_postnet, sigma=0.8), 0.01)[0, 0]
    audio = audio.cpu().numpy()
    audio = panner(audio, pan)

    write('wavfile/test.wav',
          hparams.sampling_rate, audio)
