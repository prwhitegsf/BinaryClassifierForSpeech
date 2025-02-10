#import torch
import librosa

from flask import g

from matplotlib.figure import Figure
from matplotlib.colors import Normalize
from matplotlib import colormaps

import numpy as np
import os
import random, string

from math import ceil
from app.main.utils.feature_extractors import generate_random_name



def plot_waveform(wav,sr,title='Waveform', ax=None):
    fig = Figure(figsize=(3,2.25))
    
    librosa.display.waveshow(wav, sr=sr, axis='time',ax=ax)
    ax.grid(True)
    ax.set_title(title)
    ax.title.set_size(10)
    fig.tight_layout()
    return fig


def plot_spectrogram(spectro,title=None, ylabel='Hz', ax=None):
    fig = Figure(figsize=(4,3))
    
    if ax is None:
        ax = fig.subplots()
    
    if title is not None:
        ax.set_title(title)
        ax.title.set_size(10)
    
    ax.set_ylabel(ylabel)
    ax.set_ylim(0,8000)
    
    return librosa.display.specshow(spectro, y_axis='linear', x_axis='time',ax=ax,cmap=colormaps['seismic'])



def plot_mel(spectro,title=None, ylabel='Hz', ax=None):
    fig = Figure(figsize=(4,3))
    
    if ax is None:
        ax = fig.subplots()
    
    if title is not None:
        ax.set_title(title)
        ax.title.set_size(10)
    
    ax.set_ylabel(ylabel)
    
    return librosa.display.specshow(librosa.power_to_db(spectro, ref=np.mean),x_axis='time',
                                                        y_axis='mel',norm=Normalize(vmin=-20,vmax=20), ax=ax,cmap=colormaps['seismic'])
   

def plot_mfcc(mfcc, title=None, ylabel="frequency bin", ax=None):
    fig = Figure(figsize=(4,3))
    
    if ax is None:
        ax = fig.subplots(1, 1)
    
    if title is not None:
        ax.set_title(title)
        ax.title.set_size(10)
    ax.set_ylabel(ylabel)
    return librosa.display.specshow(mfcc,norm=Normalize(vmin=-30,vmax=30), x_axis='time',ax=ax,cmap=colormaps['seismic'])

# used to plot either mel or mfcc 
def plot_feature(feature_arr, title=None, ax=None, feature_type='mfcc'):
    fig = Figure(figsize=(4,3))
    
    if ax is None:
        ax = fig.subplots(1, 1)
    
    if title is not None:
        ax.set_title(title)
        ax.title.set_size(10)
    
    if feature_type == 'mel':
        return librosa.display.specshow(librosa.power_to_db(feature_arr, ref=np.mean),x_axis='time',
                                                        y_axis='mel',norm=Normalize(vmin=-20,vmax=20), ax=ax,cmap=colormaps['seismic'])
    
    return librosa.display.specshow(feature_arr,norm=Normalize(vmin=-30,vmax=30), x_axis='time',ax=ax,cmap=colormaps['seismic'])


# We use this in the feature extractor section
def get_feature_extraction_plots(af,sess):
    
    fig = Figure(figsize=(4, 8),layout='constrained')
    wav, sr = af.get_audio()

    axs = fig.subplots(4)

    #wav
    wav = plot_waveform(wav, sr,ax=axs[0])
    axs[0].tick_params(axis='y',labelsize=7)
    axs[0].tick_params(axis='x',labelsize=7)
    axs[0].set_xlabel('')

    #spectrogram
    spectro = af.get_spectrogram()
    specplot =plot_spectrogram(spectro,title='Spectrogram',ax=axs[1])
    
    axs[1].set_title('Spectrogram')
    axs[1].title.set_size(10)
    axs[1].tick_params(axis='y',labelsize=7)
    
    axs[1].tick_params(axis='x',labelsize=7)
    axs[1].set_xlabel('')
    # spec colorbar
    spec_cb = fig.colorbar(specplot,format='%+2.0f dB')
    spec_cb.ax.tick_params(axis='y',labelsize=7)
  
    # mel
    mel = af.get_mel_spectrogram()
    melplot = plot_mel(mel, title='Mel Spectrogram',ax=axs[2])
    axs[2].tick_params(axis='y',labelsize=7)
    axs[2].tick_params(axis='x',labelsize=7)
    axs[2].set_xlabel('')
    # mel colorbar
    mel_cb = fig.colorbar(melplot,format='%+2.0f dB')
    mel_cb.ax.tick_params(axis='y',labelsize=7)
    
    # mfcc
    mfcc = af.get_mfcc()
    mfccplot = plot_mfcc(mfcc,title='MFCC',ax=axs[3])
    axs[3].tick_params(axis='y',labelsize=7)
    axs[3].tick_params(axis='x',labelsize=7)
    axs[3].set_xlabel('')
    axs[3].set_ylabel('MFC Coefficient')
    # mfcc colorbar
    mfcc_cb = fig.colorbar(mfccplot,format='%+2.0f')
    mfcc_cb.ax.tick_params(axis='y',labelsize=7)

    if sess['prev_img_file'] != None:
        os.remove('app/static/img/' + sess['prev_img_file']+'.png')  
         
    fname = generate_random_name()
    
    fig.savefig(f'app/static/img/{fname}.png',format='png')
    sess['prev_img_file'] = fname
    
    return f'img/{fname}.png'


def get_feature_plots(sess, af):
    
    fig = Figure(figsize=(4, 8),layout='constrained')
    
    feature_arr, ids = af.get_feature_group_from_npy(sess)
    plt_count = len(feature_arr)
    
    i = j = k = 0

    def add_feature_plot(feature_arr,k, ax, cb_rows=3):
        feature_plot=plot_feature(feature_arr[k],title=f'ID: {ids[k]}', ax=ax, feature_type=g.feature_type)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks(ticks=[])
        ax.set_yticks(ticks=[])

        # add colorbars to right side only
        if k > cb_rows:
            colorbar = fig.colorbar(feature_plot,format='%+2.0f')
            colorbar.ax.tick_params(axis='y',labelsize=7)


    # depending on the number of results, we format the output differently
    if plt_count < 2:

        axs=fig.subplots()
        feature_plot=plot_feature(feature_arr[k],title=f'ID: {ids[k]}', ax=axs, feature_type=g.feature_type)
        colorbar = fig.colorbar(feature_plot,format='%+2.0f')
        colorbar.ax.tick_params(axis='y',labelsize=7)
    
    elif plt_count <= 4:
        
        fig.set_figheight((plt_count*2))
        axs=fig.subplots(plt_count)
        for i in range(plt_count):
            feature_plot=plot_feature(feature_arr[k],title=f'ID: {ids[k]}', ax=axs[i], feature_type=g.feature_type)
            axs[i].set_xlabel('')
            axs[i].set_xticks(ticks=[])
            colorbar = fig.colorbar(feature_plot,format='%+2.0f')
            colorbar.ax.tick_params(axis='y',labelsize=7)

    elif plt_count <= 6:
       
        fig.set_figheight(6)
        fig.set_figwidth(6)
        
        cols = 2
        rows = ceil(plt_count/2)
        spec = fig.add_gridspec(rows,cols)
        
        for i in range(cols):
            for j in range(rows):
                if k >= plt_count:
                    break
                
                ax=fig.add_subplot(spec[j,i])
                add_feature_plot(feature_arr=feature_arr,k=k,ax=ax,cb_rows=2)
                k+=1     
    else:

        fig.set_figwidth(6)
        cols = 2
        rows = ceil(plt_count/2)
        spec = fig.add_gridspec(rows,cols)
       
        for i in range(cols):
            for j in range(rows):
                if k >= plt_count:
                    break
                ax=fig.add_subplot(spec[j,i])
                add_feature_plot(feature_arr=feature_arr,k=k,ax=ax,cb_rows=3)

                k+=1

    # remove previous image file if it exists
    if sess['prev_img_file'] != None:
        os.remove('app/static/img/' + sess['prev_img_file']+'.png')  
    
    fname = generate_random_name()

    fig.savefig(f'app/static/img/{fname}.png',format='png')
    sess['prev_img_file'] = fname
    return f'img/{fname}.png'


    
      
def show_label_distribution(record):
    
    label_count = record.record_count
    non_label_count = 1440 - label_count
   
    width = 0.25

    fig = Figure(figsize=(2, 3),layout='constrained')
    ax= fig.subplots()
    rects= ax.bar('non-labels',non_label_count,width)
    ax.bar_label(rects, label_type='edge')

    rects= ax.bar('labels',label_count,width)
    ax.bar_label(rects, label_type='edge')

    ax.set_title("Label Distribution")
    ax.tick_params(axis='y',labelsize=7)
    ax.tick_params(axis='y',labelsize=7)
    ax.title.set_size(10)
  
    ax.set_ylim(0, 1600)

    for filename in os.listdir('app/static/img/'):
             os.remove('app/static/img/' + filename)
    
    fname = generate_random_name()

    fig.savefig(f'app/static/img/{fname}.png',format='png')
    return f'img/{fname}.png'

