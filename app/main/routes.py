from flask import render_template, request,g,flash,redirect, url_for
from flask import session as sess
from app.main import bp
from app.main.forms import DataSetFilterForm, NextRecord, NextAudioRecord,TrainAndTest

from app.main.utils.feature_extractors import AudioFeatures
import app.main.utils.visualizers as viz
from app.main.utils.session_manager import SessionManager
from app.main.utils.estimator_evaluation import EvaluateResults




@bp.route('/', methods=['GET'])
@bp.route('/index', methods=['GET'])
@bp.route('/introduction',methods=['GET'])
def introduction():
     return render_template('introduction.html',
                            title_text='Exploring Simple Binary Classification for Speech')



@bp.route('/data-inspector',methods=['GET'])
def data_inspector():
     return render_template('data-inspector.html',
                            sex_distro='distro_img/actor-sex.png',
                            emotion_distro='distro_img/emotions.png',
                            phrase_distro='distro_img/phrase.png',
                            intensity_distro='distro_img/intensity.png',
                            title_text='Data Inspection')



@bp.route('/about-features',methods=['GET'])
def about_features():
    return render_template('about-features.html',
                           wav='feature_img/wave.png',
                           spectro='feature_img/spectro.png',
                           mel_spectro='feature_img/mel-spectro.png',
                           mel_filters='feature_img/mel-filters.png',
                           mfcc='feature_img/mfcc.png',
                           mfcc_process='feature_img/mfcc-flow.png',
                           title_text='Understanding Feature Extraction')




@bp.route('/feature-extractor', methods=['GET', 'POST'])
def feature_extractor():
    s = SessionManager()
    g.form = DataSetFilterForm()
    next_button = NextRecord()   
    
    if request.method == 'GET':
        s.init_sess(sess)  

    if request.method == 'POST':
        if g.form.submit.data:
            if s.check_incompatable_filters():
                flash("Incompatible filters: ")
                flash(s.flashed)
                return redirect(url_for('main.feature_extractor'))
            s.set_record_list(sess)

            
        if next_button.next.data:
            s.get_next_record(sess)

        
    af1 = AudioFeatures(g.n_mels,g.n_mfcc)
    audio_file = af1.save_audio_to_file(sess)
    img_file=viz.get_feature_extraction_plots(af1,sess)

    return render_template('feature-extractor.html',
        title='Home', 
        audio_file= audio_file,
        img_file=img_file,
        next_button=next_button, 
        record_count_text=s.message,
        record_text=s.curr_record_info,
        record_id=s.curr_id,
        title_text='Feature Extractor')



@bp.route('/about-estimators',methods=['GET'])
def about_estimators():
    return render_template('about-estimators.html',
                           hyperplanes='est_img/hyperplanes.png',
                           kernels='est_img/kernels.png',
                           conf_matrix='est_img/conf-matrix.png',
                           title_text='Estimators and measuring performance')


@bp.route('/class-selector',methods=['GET','POST'])
def class_selector():
    s = SessionManager()
    g.form = DataSetFilterForm()
    next_group = NextRecord()
    next_audio_file = NextAudioRecord()
    train_and_test = TrainAndTest()

    if request.method == 'GET':
        s.init_sess(sess)  
    
    if request.method == 'POST':
        if g.form.submit.data:
            if s.check_incompatable_filters():
                flash("Incompatible filters: ")
                flash(s.flashed)
                return redirect(url_for('main.class_selector'))
            s.set_labels_list(sess)

        elif next_group.next.data:
            s.get_next_record_group(sess)
        
        elif next_audio_file.next_audio_file.data:
            s.get_next_audio_from_group(sess)

        elif train_and_test.train_and_test.data:
            record = s.get_user_record(sess)
            if record.record_count == 1440:
                flash("Must make a selection")
                return redirect(url_for('main.class_selector'))

            return redirect(url_for('main.view_estimator_results'))
        

    af1 = AudioFeatures(g.n_mels,g.n_mfcc)
    audio_file = af1.save_audio_to_file(sess)
    img_file=viz.get_feature_plots(sess, af1)

    return render_template('class-selector.html',
        title='Home', 
        audio_file= audio_file,
        img_file=img_file,
        next_group=next_group,
        next_audio_file=next_audio_file,
        train_and_test=train_and_test,
        record_count_text=s.message, 
        record_text=s.curr_record_info,
        record_id=s.curr_id,
        title_text='Class Selector')


@bp.route('/estimator-results',methods=['GET','POST'])
def view_estimator_results():
    sm = SessionManager()
    record = sm.get_user_record(sess)
    sm.clean_audio_folder(sess)
    # check that there's valid session information, redirect to label-mfccs if not
    er = EvaluateResults()

    if request.method == 'GET':
        er.make_feature_and_label_arrays(record)
        er.scale_features()
        er.encode_labels()
        er.split_dataset()

        svc_conf_matrix = er.get_SVC_matrix()
        linsvc_conf_matrix = er.get_SVC_Linear_matrix()
        knn_conf_matrix = er.get_KNN_matrix()

        img_file = viz.show_label_distribution(record)
        label_info=er.make_label_distribution_labels(record)

        


    return render_template('estimator-results.html',
                           img_file=img_file,
                           label_info=label_info,
                           svc_scores=er.get_scores_from_conf_matrix(svc_conf_matrix),
                           linsvc_scores=er.get_scores_from_conf_matrix(linsvc_conf_matrix),
                           knn_scores=er.get_scores_from_conf_matrix(knn_conf_matrix),
                           svc_res=er.get_results_from_conf_matrix(svc_conf_matrix),
                           linsvc_res=er.get_results_from_conf_matrix(linsvc_conf_matrix),
                           knn_res=er.get_results_from_conf_matrix(knn_conf_matrix),
                           title_text='Estimator Results')


@bp.route('/observation',methods=['GET'])
def observations():
    return render_template('observations.html',
                            title_text='Observations and Additional Resources')