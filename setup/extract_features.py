import sqlalchemy as sa
from sqlalchemy.orm import Session
import os
import librosa
from setup.modules.setup_modules import get_engine
import numpy as np
import pandas as pd

from app.main.models import Ravdess as rm

def get_mfcc(waveform, *,n_fft=512, hop_length=128, n_mels=128,n_mfcc=40,sample_rate=16000):
    
    return librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=n_mfcc,
                                n_mels=n_mels,hop_length=hop_length,n_fft=n_fft,
                                htk=True,center=True) 
    
def get_mel_spectrogram(wav,n_mels):
    melspectrogram=librosa.feature.melspectrogram(y=wav, sr=16000, n_mels=n_mels,fmax=8000)
    return melspectrogram




def write_mfcc_to_npy(file_list, mels, mfcc):

    for n_mels in mels:
        for n_mfcc in mfcc:
            if n_mfcc >= n_mels:
                continue
            ds = []
            for fp in file_list:
                waveform, sample_rate = librosa.load(fp[0],sr=None)
                t_mfcc = get_mfcc(waveform, n_mels=n_mels,n_mfcc=n_mfcc)[1:]
               
                ds.append((np.mean(t_mfcc.T,axis=0),t_mfcc,fp[1]))
            
            df = pd.DataFrame(ds, columns=['features','feature_viz','id'])
            print(f'n_mels: {n_mels}   n_mfcc: {n_mfcc}')
            with open(f'datasets/RAVDESS/features/mfcc/ravdess_{n_mels}_{n_mfcc}.npy', 'wb') as f:
                np.save(f, np.array(df))


def write_mels_to_npy(file_list, mels):
    for n_mels in mels:
            ds = []
            for fp in file_list:
                waveform, sample_rate = librosa.load(fp[0],sr=None)
                t_mel = get_mel_spectrogram(waveform, n_mels=n_mels)
               
                ds.append((np.mean(t_mel.T,axis=0),t_mel,fp[1]))
            
            df = pd.DataFrame(ds, columns=['features','feature_viz','id'])
            print(f'n_mels: {n_mels}')
            with open(f'datasets/RAVDESS/features/mel/ravdess_{n_mels}.npy', 'wb') as f:
                np.save(f, np.array(df))



dbname = 'app'

basedir = os.path.abspath(os.path.dirname(__file__))
engine = get_engine(dbname)
session = Session(engine)
stmt = sa.select(rm).order_by(rm.id)
file_list = [(rec.filepath,rec.id) for rec in session.scalars(stmt)]

mels = [256, 128, 64, 53]
mfcc = [52, 39, 26, 13]

write_mfcc_to_npy(file_list,mels,mfcc)
write_mels_to_npy(file_list, mels)

