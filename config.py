import os
from dotenv import load_dotenv


basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))


class Config:
    
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'
    db_uri = os.environ.get('DATABASE_URL')
    SQLALCHEMY_DATABASE_URI = db_uri + 'app'


    


class TestConfig:
    
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'
    SQLALCHEMY_DATABASE_URI = os.environ.get('TEST_DATABASE_URL')

    TESTING = True