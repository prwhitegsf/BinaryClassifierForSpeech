from sqlalchemy import create_engine, delete
from sqlalchemy.orm import Session
from sqlalchemy.sql import func
from sqlalchemy_utils import database_exists, create_database
from app.main.models import User

import os
from dotenv import load_dotenv


basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))
db_uri = os.environ.get('DATABASE_URL')

def get_engine(dbname):
        
        engine = create_engine(db_uri+dbname)
        if not database_exists(engine.url):
            create_database(engine.url)
        return engine


 

def remove_users_by_time(dbname, expiry_minutes):
    users=[]
    with Session(get_engine(dbname)) as sess:

        rows= sess.query(func.now()-User.last_access,User.username).all()
        for row in rows:
            minutes = row[0].total_seconds() // 60
            print(minutes)
            if minutes > expiry_minutes:
                users.append(row.username)

        sess.execute(delete(User).where(User.username.in_(users)))
        sess.commit()
        return len(users)
 
def remove_all_users(dbname):
    with Session(get_engine(dbname)) as sess:

        rows_removed= sess.query(User).delete()
        sess.commit()
        return rows_removed