import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import numpy as np

from flask import g
from matplotlib.figure import Figure


class EvaluateResults():

    def __init__(self):
        # add vars for label distribution text
        # the function for that basically takes any form setting that't not 'all'
        self.features = []
        self.labels = []
        self.X_train= []
        self.X_test= []
        self.y_train= []
        self.y_test = []
        self.label_ids=[]
        self.label_desc=""


    def make_label_distribution_labels(self,record):

        label_string = ['Labeled records: ']

        if record.filters['actor'] != 'all':
            label_string.append(f'actor: {record.filters['actor']} ')

        if record.filters['sex'] != 'all':
             label_string.append(f'sex: {record.filters['sex']} ')

        if record.filters['statement'] != 'all':
            label_string.append(f'statement: {record.filters['statement']} ')

        if len(record.filters['emotion']) != 0 and record.filters['emotion'][0] != 'all':
            em = [emo for emo in record.filters['emotion']]
            label_string.append(f'emotion(s): {em} ')

        if record.filters['intensity'] != 'all':
             label_string.append(f'intensity: {record.filters['intensity'] } ')

        label_string.append(record.filters['feature_type'])

        label_string.append(f'mels: {record.filters['num_mels']}')
        if record.filters['feature_type'] == 'mfcc':
            label_string.append(f'mfccs: {record.filters['num_mfcc']}')

        return label_string


    def _load_np(self, record):
        
        n_mels= int(record.filters['num_mels'])
        n_mfcc = int(record.filters['num_mfcc'])
        feature_type = record.filters['feature_type']
        self.label_ids = record.ids
        np_path = ''
        
        if feature_type == 'mfcc':
            np_path = f'datasets/RAVDESS/features/mfcc/ravdess_{n_mels}_{n_mfcc}.npy'
        else:
            np_path = f'datasets/RAVDESS/features/mel/ravdess_{n_mels}.npy'

        return np.load(np_path, allow_pickle=True)
    
    def _apply_labels_to_np(self,record):
        ds = self._load_np(record)
        df = pd.DataFrame(ds,columns=['features','feature_viz','id'])
        df['label'] = df['id'].isin(self.label_ids)
        return df


    def make_feature_and_label_arrays(self,record):

        df = self._apply_labels_to_np(record)
        if 'label' in df.columns:
            self.features = np.vstack(df['features'])
            self.labels = np.array(df['label'])

        
    def scale_features(self):
        self.features = StandardScaler().fit_transform(self.features)

    def encode_labels(self):
        self.labels = preprocessing.LabelEncoder().fit_transform(self.labels)
    
    def split_dataset(self, test_size=0.4):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.features,
                                                                                self.labels,
                                                                                test_size=test_size)
        

    def get_SVC_matrix(self):
        model = SVC(C = 1.0,
                    gamma='auto',
                    class_weight='balanced',
                    kernel='rbf')

        model.fit(self.X_train,self.y_train)
        return self.get_conf_matrix(model)


    def get_SVC_Linear_matrix(self):
        
        model = SVC(C = 1.0,
                    class_weight='balanced',
                    kernel='linear')
        
        model.fit(self.X_train,self.y_train)
        return self.get_conf_matrix(model)
        

    def get_KNN_matrix(self):
        model = KNeighborsClassifier(n_neighbors=3,
                                    weights='distance',
                                    algorithm='ball_tree',
                                    leaf_size=10,
                                    p=2)
        
        model.fit(self.X_train,self.y_train)
        return self.get_conf_matrix(model)


    def get_conf_matrix(self, model):
        predictions = model.predict(self.X_test)
        ground_truth = self.y_test
        return confusion_matrix(ground_truth, predictions)


    def get_results_from_conf_matrix(self, conf_matrix):
        tn, fp, fn, tp = conf_matrix.ravel()
        return [str(round(tn,2)), str(round(fp,2)),str(round(fn,2)),str(round(tp,2))]
    
    
    def get_scores_from_conf_matrix(self, conf_matrix):
        tn, fp, fn, tp = conf_matrix.ravel()

        precision = tp / (tp +fp) if (tp +fp) != 0 else 0
        recall = tp / (tp +fn) if (tp +fn) != 0 else 0
        accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) != 0 else 0
        scores  = [str(round(recall,2)), str(round(precision,2)),str(round(accuracy,2))]
        return scores

    