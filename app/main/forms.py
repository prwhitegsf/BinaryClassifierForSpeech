from flask_wtf import FlaskForm
from wtforms import SubmitField, RadioField, SelectField, SelectMultipleField



def create_range():
        rng = list(range(1,25))
        nums = ['all']
        for i in rng:
             item = str(i)
             nums.append(item)      
        
        return nums

class ViewDataDistributions(FlaskForm):

    submit = SubmitField('Submit')
    
    select = SelectField('Data Distributions', choices=[
        ('actors'),
        ('actor-sex'),
        ('statement'),
        ('intensity'),
        ('emotion')], default='emotion')
    
    

class DataSetFilterForm(FlaskForm):
    

    submit = SubmitField('Submit')
    actor = SelectField('Actor Number', choices=create_range(),default='all')
    sex  = SelectField('Sex',choices=['all','female','male'],default='all')
    statement = SelectField('Statement',choices=['all','1','2'],default='all')
    emotion =  SelectMultipleField('Emotion', choices=[('all','all'),
                                                        ('neutral','neutral'),
                                                        ('calm','calm'),
                                                        ('happy','happy'),
                                                        ('sad','sad'),
                                                        ('angry','angry'),
                                                        ('fearful','fearful'),
                                                        ('disgust','disgust'),
                                                        ('surprised','surprised')],
                                                        default=('all','all'))

    intensity = SelectField('Intensity',choices=['all','1','2'],default='all')

    feature_type = SelectField('Feature Type',choices=['mel','mfcc'],default='mfcc')

    num_mels = RadioField("Mel Filter Count", choices=[('256','256'),
                                                            ('128','128'),
                                                               ('64','64'),
                                                               ('53','53')],
                                                               default='128')
    
    num_mfcc = RadioField("MFCC Count", choices=[ ('52','52'),
                                                   ('39','39'),
                                                   ('26','26'),
                                                   ('13','13')],
                                                   default='39')
    


class NextRecord(FlaskForm):
     next = SubmitField('Next')



class NextAudioRecord(FlaskForm):
     next_audio_file = SubmitField('Next Audio File')


class TrainAndTest(FlaskForm):
     train_and_test = SubmitField('Train and Test')