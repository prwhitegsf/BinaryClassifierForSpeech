import logging
from logging.handlers import RotatingFileHandler

import os
from flask import Flask


from config import Config
from app.main.models import db




def create_app(config_class=Config):
    
    # App object from config class
    app = Flask(__name__)
    app.config.from_object(config_class)

    db.init_app(app)
     
    from app.main import bp as main_bp
    app.register_blueprint(main_bp) 


    if not app.debug and not app.testing:

        if not os.path.exists('logs'):
            os.mkdir('logs')
        file_handler = RotatingFileHandler('logs/test.log',
                                        maxBytes=10240, backupCount=10)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s '
            '[in %(pathname)s:%(lineno)d]'))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)

        app.logger.setLevel(logging.INFO)
        app.logger.info('test startup')


    return app