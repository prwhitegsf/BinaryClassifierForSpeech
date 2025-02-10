from sqlalchemy import create_engine
from sqlalchemy.orm import Session
import os, glob
import sys
from sqlalchemy_utils import database_exists, create_database
from setup.modules.setup_modules import get_engine
from app.main.models import Ravdess,User,Base




# we derive metadata from the filename according to:
# https://zenodo.org/records/1188976
class CreateRAVDESSMetadata():

    def __init__(self, db_name='db_name', folder='datasets/RAVDESS/audio/'):
        
        self.emotions = {
            '01':'neutral',
            '02':'calm',
            '03':'happy',
            '04':'sad',
            '05':'angry',
            '06':'fearful',
            '07':'disgust',
            '08':'surprised'
        }

        self.dataset_folder = folder
        self.engine = get_engine(db_name)
        self.metadata_obj = Base.metadata.create_all(self.engine)
        
    def get_actor(self, filename):
        return int(filename.split("-")[6].split('.')[0]) 
        
    def get_actor_sex(self,filename):
        if int(filename.split("-")[6].split('.')[0]) % 2 == 0: return 'female'
        else: return 'male'

    def get_id(self, filename):
        fn_arr = filename.split('-')
        actor = fn_arr.pop(-1).split('.')[0]
        id_str = ''.join([val[1] for val in fn_arr])
        id_str=actor+id_str

        return int(id_str)


    def get_md(self):
        session = Session(self.engine)
        count = 1
        
        for file in glob.glob(f'{self.dataset_folder}Actor_*/*.wav'):
            
            file = os.path.normpath(file)
            filename = os.path.basename(file)
            

            rec = Ravdess(id=self.get_id(filename),
                          filepath=file, 
                          actor=int(filename.split("-")[6].split('.')[0]) ,
                          sex=self.get_actor_sex(filename),
                          statement=int(filename.split("-")[4]),
                          emotion=self.emotions[filename.split("-")[2]],
                          intensity = int(filename.split("-")[3]),
                          sample_rate=16000,
                          filesize=os.path.getsize(file))
      
            session.add(rec)
            count += 1

        session.commit()

   

dbname = 'app'

if len(sys.argv) > 1:
    dbname=sys.argv[1]

md = CreateRAVDESSMetadata(db_name=dbname)
md.get_md()
