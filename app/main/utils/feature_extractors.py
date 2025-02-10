
import librosa
import soundfile
import os
import random, string
import numpy as np
import pandas as pd
from flask import g

def generate_random_name(len=5):
     letters = string.ascii_lowercase
     name = ''.join(random.choice(letters) for i in range(len))
     return name


class AudioFeatures():

    def __init__(self, n_mels=128, n_mfcc=40):
        #fp = sess['record_list'][sess['curr_record']] 
        fp=g.fp
        self.wav, self.sr = librosa.load(fp)
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
       


    def change_audio_file(self, fp):
        self.wav, self.sr = librosa.load(fp)

    def get_spectrogram(self):
        spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(self.wav, n_fft=512,hop_length=512)), ref=np.max)
        #print("spectrogram: ",spectrogram.shape)
        return spectrogram
    
    def get_mel_spectrogram(self):
        melspectrogram=librosa.feature.melspectrogram(y=self.wav, sr=self.sr, n_mels=self.n_mels,fmax=8000)
        #print("mel spectrogram: ",melspectrogram.shape)
        return melspectrogram

    def get_mfcc(self,num_filters=40):
        # Compute the MFCCs for all STFT frames and get the mean of each column of the resulting matrix to create a feature array
        # 40 filterbanks = 40 coefficients
        mfc_coefficients= librosa.feature.mfcc(y=self.wav, sr=self.sr, n_mfcc=num_filters,n_fft=512)
        #print("mfcc: ",mfc_coefficients.shape)
        return mfc_coefficients[1:]
 

    def get_audio(self):
        return self.wav, self.sr
    
# This shouldn't remove every file in static/audio, it should only remove the last file 
# Meaning the file name should be recorded into the db and with each "next" press, we delete that file and save a new one.
# Alternately, we could possible assign a random name for all audio files for a single user, then clean them out when we clean users    
    def save_audio_to_file(self, sess):
         # first remove prev record
         
         if sess['prev_audio_file'] != None:
             os.remove('app/static/audio/' + sess['prev_audio_file']+'.wav')   

         fname = generate_random_name()
         
         soundfile.write(f'app/static/audio/{fname}.wav', self.wav, self.sr, format="wav")
         sess['prev_audio_file'] = fname
         return f'audio/{fname}.wav'




    def _load_initial_data(self):
        np_path =''
        if g.feature_type == 'mfcc':
            np_path = f'datasets/RAVDESS/features/mfcc/ravdess_{self.n_mels}_{self.n_mfcc}.npy'
        else:
            np_path = f'datasets/RAVDESS/features/mel/ravdess_{self.n_mels}.npy'
        
        ds = np.load(np_path, allow_pickle=True)
        return pd.DataFrame(ds,columns=['features','feature_viz','id'])


    def get_feature_group_from_npy(self, sess):
        df = self._load_initial_data()
        feature_arr =[]
        ids = []
        for id in g.id_group:
            
            ids.append(id)
            label_df = df.loc[df['id'] == id]
            feature_arr.append(label_df['feature_viz'].iloc[0])
        return feature_arr, ids

