import sqlalchemy as sa
from sqlalchemy.orm import Session
import os
from app.main.models import Ravdess as rm

from app.main.utils.estimator_evaluation import EvaluateResults


from setup.modules.setup_modules import get_engine
import numpy as np
import pandas as pd

def create_where_clause(filters):

    stmt = 'ravdess.id >= 0'

    if filters['actor'] != 'all':
        stmt += f' AND ravdess.actor = {filters['actor']}'

    if filters['sex'] != 'all':
        stmt += f' AND ravdess.sex = \'{filters['sex']}\''

    if filters['statement'] != 'all':
        stmt += f' AND ravdess.statement = {filters['statement']}'

    if len(filters['emotion']) != 0 and filters['emotion'][0] != 'all':

        em = [emo for emo in filters['emotion']]
        em.append("-")
        
        stmt+= f' AND ravdess.emotion in {tuple(em)}'

    if filters['intensity'] != 'all':
        stmt+= f' AND ravdess.intensity = {filters['intensity']}'
    print(stmt)
    return stmt





def create_record_list(stmt):
    urls=[]
    ids=[]
    for row in stmt:
        urls.append(row.filepath)
        ids.append(row.id)
    return ids


def get_filtered_records(session,filters):
    stmt = session.execute(sa.select(rm).where(sa.text(create_where_clause(filters)))).scalars()  
    
    return create_record_list(stmt)


class Mock_Record():
    
    def __init__(self):

    
        self.filters = {
            'actor' : 'all',
            'sex'   : 'all',
            'statement' : 'all',
            'intensity' : 'all',
            'emotion' : ['angry'],
            'num_mels'   : '128',
            'num_mfcc'   : '39',
            'feature_type' : 'mfcc'
        }

        self.ids = []

    def get_filter_values_as_string(self):
        
        vals = list(self.filters.values())
        str_out = ''
        for val in vals:
            str_out += str(val)+","


        return str_out

    


# Setup db coms

dbname = 'app'

basedir = os.path.abspath(os.path.dirname(__file__))
engine = get_engine(dbname)
session = Session(engine)
stmt = sa.select(rm).order_by(rm.id)

er = EvaluateResults()


record = Mock_Record()

actors = ['all']#,'1','2','3','4','5','6','7','8','9','10','11','12']
sexes = ['all']#,'male','female']
phrases = ['all']#,'1','2']
intensities = ['all']#,'1','2']
emotions = [['neutral']]#,['calm'],['happy'],['sad'],['angry'],['fearful'],['disgust'],['surprised']]
feature_types = ['mfcc','mel']
mels = ['256','128','64']
mfccs = ['52','39','26','13']

for actor in actors:
    record.filters['actor'] = actor
    for sex in sexes:
        record.filters['sex'] = sex
        for phrase in phrases:
            record.filters['statement'] = phrase
            for intensity in intensities:
                record.filters['intensity'] = intensity
                for emotion in emotions:
                    record.filters['emotion'] = emotion 
                    for feature_type in feature_types:
                        record.filters['feature_type'] = feature_type
                        for mel in mels:
                            record.filters['num_mels'] = mel
                            for mfcc in mfccs:
                                record.filters['num_mfcc'] = mfcc
        
                                record.ids = get_filtered_records(session,record.filters)

                                if len(record.ids) != 0:
                                    #print(record.ids)
                                    er.make_feature_and_label_arrays(record)
                                    er.scale_features()
                                    er.encode_labels()
                                    er.split_dataset()

                                    svc_conf_matrix = er.get_SVC_matrix(record)
                                    linsvc_conf_matrix = er.get_SVC_Linear_matrix(record)
                                    knn_conf_matrix = er.get_KNN_matrix(record)


                                    svc_res=er.get_scores_from_conf_matrix(svc_conf_matrix)
                                    linsvc_res=er.get_scores_from_conf_matrix(linsvc_conf_matrix)
                                    knn_res=er.get_scores_from_conf_matrix(knn_conf_matrix)

                                    # actor, sex, statement, intensity, emotion, mels, mfcc, feature_type,rbf-recall, rbf-prec, rbf-accur, lin-recall, lin-prec, lin-acc, knn-recall, knn-prec, knn-acc

                                    svc = ','.join([result for result in svc_res])
                                    linsvc = ','.join([result for result in linsvc_res])
                                    knn = ','.join([result for result in knn_res])

                                    line = record.get_filter_values_as_string()+svc+","+linsvc+","+knn+'\n'

                                    with open('auto-results.txt','a') as file:
                                            file.write(line)