### THESE ARE ONLY SUGGESTED IMPORTS ###

# web (db and server) imports
from flask import Flask, render_template, request, url_for, jsonify, make_response
import pymysql
from pymongo import MongoClient
# machine learning imports
from sklearn import model_selection, preprocessing, naive_bayes, metrics, ensemble, neighbors
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
# helpers
from collections import Counter
from datetime import datetime
from time import *
# need pickle to store (if you want) binary files in mongo
import pickle
# json/bson handling
import json
from bson import ObjectId
import pandas as pd


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        return json.JSONEncoder.default(self, o)


def get_current_time():
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    return dt_string


# mysql connection
password = 'password'
# Connect to the database
connection = pymysql.connect(host='csmysql.cs.cf.ac.uk',
                             user='c2090580',
                             password=password,
                             db='c2090580_assessment_1',
                             charset='utf8mb4',
                             )
cursor = connection.cursor(cursor=pymysql.cursors.DictCursor)


#mongodb connection
client = MongoClient('mongodb://c2090580:password@csmongo.cs.cf.ac.uk:27017/c2090580',ssl=True)
mongodb = client['c2090580']
collection = mongodb['collection']

# client = MongoClient('mongodb://localhost/27017')
# mongodb = client['assessment_2']
# collection = mongodb['collection']




app = Flask(__name__)




def reset_views():
    sql_1 = """
    drop view if exists training_data_view ;

    """
    sql_2 = """
    drop view if exists test_data_view;
    """
    result_1 = cursor.execute(sql_1)
    result_2 = cursor.execute(sql_2)
    print('--------------')
    print('successfully dropped training_data and test_data views')


def create_training_view():
    sql = """
    create view training_data_view as
    select concat(selftext, post_title) text,subreddit_name label
    from posts_info
    join subreddit_info si on si.subreddit_id = posts_info.post_subreddit_id
    limit 15000
    """
    result = cursor.execute(sql)
    print('--------------')
    print('successfully created training set')
def create_test_view():

    sql = """
    create view test_data_view as
    select concat(selftext, post_title) text,subreddit_name label
    from posts_info
    join subreddit_info si on si.subreddit_id = posts_info.post_subreddit_id
    limit 15000,19940
    """
    result = cursor.execute(sql)
    print('--------------')
    print('successfully created text set')


def check_views():
    sql_1 = """
    select count(*) count from training_data_view;

    """
    sql_2 = """
    select count(*) count from test_data_view;

    """
    result_1 = cursor.execute(sql_1)
    x = []
    for i in cursor.fetchall():
        x.append(i)
    result_2 = cursor.execute(sql_2)
    y = []
    for j in cursor.fetchall():
        y.append(j)
    print('--------------')
    print('training data size:', x)
    print('test data size:', y)



@app.route('/')
def form():
    reset_views()
    create_training_view()
    create_test_view()
    check_views()
    return render_template('index.html')


training_data_x = None
test_data_x = None
training_data_y = None
test_data_y = None
training_data_count = None
test_data_count = None
count_vectorizer = None

@app.route('/experiment', methods=['GET', 'POST'])
def experiment():
    sql = """
        select text, label
        from training_data_view;
        """
    result = cursor.execute(sql)

    data_frame_training = []
    for i in cursor.fetchall():
        data_frame_training.append(i)

    data_frame_training = pd.DataFrame(data_frame_training)

    sql_test = """
        select text, label
        from test_data_view;
    """
    result_test = cursor.execute(sql_test)
    data_frame_test = []
    for j in cursor.fetchall():
        data_frame_test.append(j)
    data_frame_test = pd.DataFrame(data_frame_test)


    print('--------------')

    print('loading, maybe need 1 or 2 mins')

    data_frame_training['label'] = data_frame_training['label'].apply(
        lambda x: 1 if (x.lower().find('covid') != -1 or x.lower().find('coronavirus') != -1) else 0
    ).astype(int)

    data_frame_test['label'] = data_frame_test['label'].apply(
        lambda x: 1 if (x.lower().find('covid') != -1 or x.lower().find('coronavirus') != -1) else 0
    ).astype(int)

    data_frame_training = data_frame_training.sample(frac=1)
    data_frame_test = data_frame_test.sample(frac=1)
    global training_data_x, test_data_x, training_data_y, test_data_y, count_vectorizer, training_data_count, test_data_count

    training_data_x = data_frame_training['text']
    test_data_x = data_frame_test['text']
    training_data_y = data_frame_training['label']
    test_data_y = data_frame_test['label']

    # print('-------')
    # print(training_data_x)
    # print('-------')
    # print(test_data_x)
    # print('-------')
    # print(training_data_y)
    # print('-------')
    # print(test_data_y)
    # print('-------')

    count_vectorizer = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
    count_vectorizer.fit(data_frame_training['text'])
    training_data_count = count_vectorizer.transform(training_data_x)
    test_data_count = count_vectorizer.transform(test_data_x)


    global result_logistic_regression, result_random_forest, result_svm_classifier, result_decision_tree_classifier, result_naive_bayes,result_knn
    global logistic_regression_model_time, random_forest_model_time, svm_classifier_model_time, naive_bayes_model_time, decision_tree_classifier_model_time,knn_model_time




    print('loading linear classifier')
    # Linear Classifier
    begin_time = time()
    logistic_regression_model = LogisticRegression(max_iter=2000)
    logistic_regression_model.fit(training_data_count, training_data_y)
    predictions = logistic_regression_model.predict(test_data_count)
    result_logistic_regression = metrics.accuracy_score(predictions, test_data_y)
    end_time = time()
    logistic_regression_model_time = end_time - begin_time

    #
    # print('loading random forest model')
    # # RandomForest
    # begin_time = time()
    # random_forest_model = ensemble.RandomForestClassifier()
    # random_forest_model.fit(training_data_count, training_data_y)
    # predictions = random_forest_model.predict(test_data_count)
    # result_random_forest = metrics.accuracy_score(predictions, test_data_y)
    # end_time = time()
    # random_forest_model_time = end_time - begin_time

    print('loading svm classifier')
    # SVM Classifier
    begin_time = time()
    svm_classifier_model = SVC()
    svm_classifier_model.fit(training_data_count, training_data_y)
    predictions = svm_classifier_model.predict(test_data_count)
    result_svm_classifier = metrics.accuracy_score(predictions, test_data_y)
    end_time = time()
    svm_classifier_model_time = end_time - begin_time


    print('loading naive bayes')
    # naive bayes
    begin_time = time()
    naive_bayes_model = naive_bayes.MultinomialNB()
    naive_bayes_model.fit(training_data_count,training_data_y)
    predictions = svm_classifier_model.predict(test_data_count)
    result_naive_bayes = metrics.accuracy_score(predictions, test_data_y)
    end_time = time()
    naive_bayes_model_time = end_time - begin_time



    print('loading decision tree classifier')
    # decision tree classifier
    begin_time = time()
    decision_tree_classifier_model = DecisionTreeClassifier(max_depth=2000)
    decision_tree_classifier_model.fit(training_data_count,training_data_y)
    predictions = decision_tree_classifier_model.predict(test_data_count)
    result_decision_tree_classifier = metrics.accuracy_score(predictions, test_data_y)
    end_time = time()
    decision_tree_classifier_model_time = end_time - begin_time



    # print('loading knn classifier')
    # #KNN
    # begin_time = time()
    # knn_model = neighbors.KNeighborsClassifier()
    # knn_model.fit(training_data_count,training_data_y)
    # predictions = knn_model.predict(test_data_count)
    # result_knn = metrics.accuracy_score(predictions, test_data_y)
    # end_time = time()
    # knn_model_time = end_time - begin_time


    logistic_regression_binaries = pickle.dumps(logistic_regression_model)
    svm_classifier_binaries = pickle.dumps(svm_classifier_model)
    naive_bayes_binaries = pickle.dumps(naive_bayes_model)
    # random_forest_binaries = pickle.dumps(random_forest_model)
    decision_tree_classifier_binaries = pickle.dumps(decision_tree_classifier_model)




    collection.remove()

    collection.save({'model_name': 'logistic_regression',
                     'binaires_data': logistic_regression_binaries,
                     'score': result_logistic_regression,
                     'time': logistic_regression_model_time
                    })



    # the binaires_Data of random_forest is too big
    # collection.save({'model_name': 'random_forest',
    #                  'binaires_data': random_forest_binaries,
    #                  'score': result_random_forest,
    #                  'time': random_forest_model_time
    #                 })


    collection.save({'model_name': 'svm_classifier',
                     'binaires_data': svm_classifier_binaries,
                     'score': result_svm_classifier,
                     'time': svm_classifier_model_time
                    })

    collection.save({'model_name': 'naive_bayes',
                     'binaires_data': naive_bayes_binaries,
                     'score': result_naive_bayes,
                     'time': naive_bayes_model_time
                    })

    collection.save({'model_name': 'decision_tree_classifier',
                     'binaires_data': decision_tree_classifier_binaries,
                     'score': result_decision_tree_classifier,
                     'time': decision_tree_classifier_model_time
                    })

    # collection.save({'model_name': 'knn_classifier',
    #                  'binaires_data': knn_binaries,
    #                  'score': result_knn,
    #                  'time': knn_model_time
    #                 })





    print('load model and store data successful')

    return JSONEncoder().encode("congratulations! model training finished! "+" and saved to mongodb")


@app.route('/report', methods=['GET', 'POST'])
def retrieve_results():
    best_scoring_model_top3 = collection.find({}, {"model_name": 1}).sort("score", -1).limit(3)
    fast_model_top_3 = collection.find({}, {"model_name": 1}).sort("time", 1).limit(3)
    best_model_name = " "
    print("top 3 best scoring model: ")
    for x in best_scoring_model_top3:
        best_model_name += x['model_name']
        best_model_name += '  '
        print(x)

    print("top 3 fast model: ")
    fast_model_name = ""
    for y in fast_model_top_3:
        fast_model_name += y['model_name']
        fast_model_name += '  '
        print(y)

    result = '  top 3 best scoring model:  ' + best_model_name + "----&&&----  top 3 fast model:  " + fast_model_name

    global model_loaded
    load_model = collection.find()
    print('loaded models below')
    for i in load_model:
        model_loaded = pickle.loads(i['binaires_data'])
        name = i['model_name']
        print(name)

    return JSONEncoder().encode("result :" + result)






@app.route('/submitted', methods=['POST'])
def submitted_form():
    input_text = pd.DataFrame([request.form.to_dict()['input_text']],columns=['i'])
    input_word = count_vectorizer.transform(input_text['i'])
    predictions = model_loaded.predict(input_word)
    if predictions[0] == 0:
        result = { "input_text": request.form.to_dict()['input_text'],"prediction":'not-covid' }
    else:
        result = { "input_text": request.form.to_dict()['input_text'],"prediction":'covid' }
    return jsonify(result)




if __name__ == '__main__':
	app.run(host='0.0.0.0', port=8080, debug=True)
