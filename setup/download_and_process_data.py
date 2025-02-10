from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile


folder='datasets/RAVDESS/audio/'

# From https://zenodo.org/records/1188976
zipurl = 'https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1'
with urlopen(zipurl) as zipresp:
    with ZipFile(BytesIO(zipresp.read())) as zfile:
        zfile.extractall(folder)


import os, glob
import librosa
import soundfile

folder='datasets/RAVDESS/audio/'
for file in glob.glob(f'{folder}Actor_*/*.wav'):
    file = os.path.normpath(file)
    # load only seconds 1 - 3
    waveform, sample_rate = librosa.load(file, sr=16000,offset=1.0, duration=(2.5))
    
    # save over original file
    soundfile.write(file, waveform, 16000, format="wav")