#####Basic Imports
import time
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import codecs
import csv
import re
from datetime import datetime
import numpy.random as rnd
# to make this notebook's output stable across runs
rnd.seed(42)




from string import punctuation
#####Word Vectors
from gensim import models
from gensim.models import KeyedVectors
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import LancasterStemmer
stemmer =  LancasterStemmer()
lemmer = WordNetLemmatizer()
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize, RegexpTokenizer
alpha_tokenizer = RegexpTokenizer('[A-Za-z]\w+')
#####SKLEARN
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
#####TENSORFLOW
import tensorflow as tf
from functools import partial
from IPython.display import clear_output, Image, display, HTML


##### This is the Helper File with all the function framework for building a Neural Net using Tensorflow on the Quora Question Pairs Problem







def calculate_vif_(X, thresh=5.0):
    from statsmodels.stats.outliers_influence import variance_inflation_factor 
    variables = list(X.columns)
    dropped=True
    while dropped:
        dropped=False
        vif = [variance_inflation_factor(X[variables].values, ix) for ix in range(X[variables].shape[1])]
        #print(vif)
        maxloc = vif.index(max(vif))
        
        if max(vif) > thresh:
            #print('dropping \'' + X[variables].columns[maxloc] + '\' at index: ' + str(maxloc))
            variables.pop(maxloc)
            dropped=True

    #print('Remaining variables:')
    #print(X[variables])
    return X[variables]



def scale_features_(X,round_digit=3):
    from sklearn_pandas import DataFrameMapper
    import numpy as np
    mapper = DataFrameMapper([(X.columns, StandardScaler())])
    scaled_features = np.round(mapper.fit_transform(X.copy()), round_digit)
    scaled_features_df = pd.DataFrame(scaled_features, index=X.index, columns=X.columns)



def word_vector(word_vector_file='../downloads/GoogleNews-vectors-negative300.bin'):
    wordVectorStartTime = time.time()
    print('Indexing word vectors')

    word2vec = KeyedVectors.load_word2vec_format(word_vector_file, \
        binary=True)
    print('Found %s word vectors of word2vec' % len(word2vec.vocab))
    wordVectorDurationInMinutes = (time.time()-wordVectorStartTime)/60.0
    print("feature extraction took %.2f minutes" % (wordVectorDurationInMinutes))





def text_to_wordlist(text, stop_words, remove_stop_words=True, stem_words=False):
    # Clean the text, with the option to remove stop_words and to stem words.

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"what's", "", text)
    text = re.sub(r"What's", "", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"I'm", "I am", text)
    text = re.sub(r" m ", " am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"60k", " 60000 ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e-mail", "email", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"quikly", "quickly", text)
    text = re.sub(r" usa ", " America ", text)
    text = re.sub(r" USA ", " America ", text)
    text = re.sub(r" u s ", " America ", text)
    text = re.sub(r" uk ", " England ", text)
    text = re.sub(r" UK ", " England ", text)
    text = re.sub(r"india", "India", text)
    text = re.sub(r"switzerland", "Switzerland", text)
    text = re.sub(r"china", "China", text)
    text = re.sub(r"chinese", "Chinese", text) 
    text = re.sub(r"imrovement", "improvement", text)
    text = re.sub(r"intially", "initially", text)
    text = re.sub(r"quora", "Quora", text)
    text = re.sub(r" dms ", "direct messages ", text)  
    text = re.sub(r"demonitization", "demonetization", text) 
    text = re.sub(r"actived", "active", text)
    text = re.sub(r"kms", " kilometers ", text)
    text = re.sub(r"KMs", " kilometers ", text)
    text = re.sub(r" cs ", " computer science ", text) 
    text = re.sub(r" upvotes ", " up votes ", text)
    text = re.sub(r" iPhone ", " phone ", text)
    text = re.sub(r"\0rs ", " rs ", text) 
    text = re.sub(r"calender", "calendar", text)
    text = re.sub(r"ios", "operating system", text)
    text = re.sub(r"gps", "GPS", text)
    text = re.sub(r"gst", "GST", text)
    text = re.sub(r"programing", "programming", text)
    text = re.sub(r"bestfriend", "best friend", text)
    text = re.sub(r"dna", "DNA", text)
    text = re.sub(r"III", "3", text) 
    text = re.sub(r"the US", "America", text)
    text = re.sub(r"Astrology", "astrology", text)
    text = re.sub(r"Method", "method", text)
    text = re.sub(r"Find", "find", text) 
    text = re.sub(r"banglore", "Banglore", text)
    text = re.sub(r" J K ", " JK ", text)
    
    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation])
    
    # Optionally, remove stop words
    if remove_stop_words:
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)
    
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return(text)



def process_questions(question_list, questions, question_list_name, dataframe, stop_words,stem=False):
    '''transform questions and display progress'''
    for question in questions:
        question_list.append(text_to_wordlist(question, stem_words=stem,stop_words=stop_words))
        



def clean_up(trainDF, valDF,stem=False):
    cleanUpStartTime = time.time()

    stop_words = ['the','a','an','and','but','if','or','because','as','what','which','this','that','these','those','then',
              'just','so','than','such','both','through','about','for','is','of','while','during','to','What','Which',
              'Is','If','While','This']

    train_question1 = []
    process_questions(train_question1, trainDF.question1, 'train_question1', trainDF, stem=stem, stop_words=stop_words)
    train_question2 = []
    process_questions(train_question2, trainDF.question2, 'train_question2', trainDF, stem=stem, stop_words=stop_words)

    val_question1 = []
    process_questions(val_question1, valDF.question1, 'train_question1', valDF, stem=stem, stop_words=stop_words)
    val_question2 = []
    process_questions(val_question2, valDF.question2, 'train_question2', valDF, stem=stem, stop_words=stop_words)

    cleanUpDurationInMinutes = (time.time()-cleanUpStartTime)/60.0
    print("Clean Up took %.2f minutes" % (cleanUpDurationInMinutes))
    return train_question1, train_question2, val_question1, val_question2



def BOW(train_data, val_data,train_question1, train_question2, val_question1, val_question2, max_df=0.999, min_df=50, maxNumFeatures=1000, ngrams=(1,5), analyzer='char', binary_flag=True, lowercase_flag=True, dense=1):
    #%% create dictionary and extract BOW features from questions
    featureExtractionStartTime = time.time()
    # bag of letter sequences (chars or words) -- Char Works Better
    BagOfWordsExtractor = CountVectorizer(max_df=max_df, min_df=min_df, max_features=maxNumFeatures, 
                                      analyzer=analyzer, ngram_range=ngrams, 
                                      binary=binary_flag, lowercase=lowercase_flag, stop_words='english')
    import pandas as pd
    df = pd.Series(np.array(train_question1))
    df2 = pd.Series(np.array(train_question2))
    df3 = pd.concat((df,df2))
    unique_sentences = df3.unique()
    BagOfWordsExtractor.fit(unique_sentences)
    trainQuestion1_BOW_rep = BagOfWordsExtractor.transform(np.array(train_question1))
    trainQuestion2_BOW_rep = BagOfWordsExtractor.transform(np.array(train_question2))
    valQuestion1_BOW_rep = BagOfWordsExtractor.transform(np.array(val_question1))
    valQuestion2_BOW_rep = BagOfWordsExtractor.transform(np.array(val_question2))
    train_lables = np.array(train_data.ix[:,'is_duplicate'])
    val_lables = np.array(val_data.ix[:,'is_duplicate'])


    X_train = -(trainQuestion1_BOW_rep != trainQuestion2_BOW_rep).astype(float) + trainQuestion1_BOW_rep.multiply(trainQuestion2_BOW_rep).astype(float)
    X_val = -(valQuestion1_BOW_rep != valQuestion2_BOW_rep).astype(float) + valQuestion1_BOW_rep.multiply(valQuestion2_BOW_rep).astype(float)

    y_train = train_lables
    y_val = val_lables

    if dense==1:
        X_train = X_train.toarray()
        X_val =  X_val.toarray()

    featureExtractionDurationInMinutes = (time.time()-featureExtractionStartTime)/60.0
    print("feature extraction took %.2f minutes" % (featureExtractionDurationInMinutes))
    return BagOfWordsExtractor, X_train, X_val, y_train, y_val
 

def load_train(train_file='../downloads/train.csv', na_mode='empty', test_size=0.2):
    trainDFPrime = pd.read_csv(train_file)
    trainDFPrime = trainDFPrime.fillna(na_mode)
    print(trainDFPrime.ix[:7,3:])
    print(trainDFPrime.ix[5][3])
    print(trainDFPrime.ix[5][4])
    trainDF , valDF = train_test_split(trainDFPrime, test_size = test_size)
    return trainDF, valDF




def load_test(test_file='../downloads/test.csv'):
    testDF = pd.read_csv(test_file)
    testDF.ix[testDF['question1'].isnull(),['question1','question2']] = 'random empty question'
    testDF.ix[testDF['question2'].isnull(),['question1','question2']] = 'random empty question'
    return testDF
   

def process_test(BagOfWordsExtractor,testDF, dense=1, stem=False):
    testloadStartTime = time.time()
    stop_words = ['the','a','an','and','but','if','or','because','as','what','which','this','that','these','those','then',
              'just','so','than','such','both','through','about','for','is','of','while','during','to','What','Which',
              'Is','If','While','This']


    test_question1 = []
    process_questions(test_question1, testDF.question1, 'test_question1', testDF, stem=stem, stop_words=stop_words)
    test_question2 = []
    process_questions(test_question2, testDF.question2, 'test_question2', testDF, stem=stem, stop_words=stop_words)

    testQuestion1_BOW_rep = BagOfWordsExtractor.transform(test_question1)
    testQuestion2_BOW_rep = BagOfWordsExtractor.transform(test_question2)

#X_test = -(testQuestion1_BOW_rep != testQuestion2_BOW_rep).astype(int)
    X_test = -(testQuestion1_BOW_rep != testQuestion2_BOW_rep).astype(int) + testQuestion1_BOW_rep.multiply(testQuestion2_BOW_rep)

    if dense==1:
        X_test = X_test.toarray()

    testLoadDurationInMinutes = (time.time()-testloadStartTime)/60.0
    print("Process Test %.2f minutes" % (testLoadDurationInMinutes))
    return X_test


def get_model_params():
    gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    return {gvar.op.name: value for gvar, value in zip(gvars, tf.get_default_session().run(gvars))}

def restore_model_params(model_params):
    gvar_names = list(model_params.keys())
    assign_ops = {gvar_name: tf.get_default_graph().get_operation_by_name(gvar_name + "/Assign")
                  for gvar_name in gvar_names}
    init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
    feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
    tf.get_default_session().run(assign_ops, feed_dict=feed_dict)


#def tensor(n_inputs=100, n_hidden=[500,300], n_outputs=2, momentum=0.9, learning_rate=0.1, batch_normal=1, dropout_rate=0.5, activation=tf.nn.relu, n_epochs=2, batch_size=250):
def tensor(length,X_train,y_train,X_val, y_val,n_inputs=10, n_hidden=[500,300], n_outputs=2, momentum=0.9, learning_rate=0.1, batch_normal=1, dropout_rate=0.5, activation=tf.nn.relu, n_epochs=2, batch_size=250):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    logdir = "{}/run-{}/".format(root_logdir, now)
    
    tf.reset_default_graph()
    NeuralNetStartTime = time.time()

    n_inputs = n_inputs 

    n_outputs = n_outputs
    learning_rate = learning_rate

    momentum = momentum

    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int64, shape=(None), name="y")
    is_training = tf.placeholder(tf.bool, shape=(), name='is_training')
    

    with tf.name_scope("dnn"):
        he_init = tf.contrib.layers.variance_scaling_initializer() #Don't Like Using Contrib

        my_batch_norm_layer = partial(
            tf.layers.batch_normalization,
            training=is_training,
            momentum=momentum)

        my_dense_layer = partial(
            tf.layers.dense,
        kernel_initializer=he_init, activation=activation)
        ###First Layer
        hidden1 = my_dense_layer(X, n_hidden[0], name="hidden1")
        hidden1_drop = tf.layers.dropout(hidden1, dropout_rate, training=is_training)
        if batch_normal==1:
            bn1 = tf.nn.elu(my_batch_norm_layer(hidden1_drop))
        else:
            bn1 = hidden1_drop

        ###Second Layer
        hidden2 = my_dense_layer(bn1, n_hidden[1], name="hidden2")
        hidden2_drop = tf.layers.dropout(hidden2, dropout_rate, training=is_training)
        if batch_normal==1:
            bn2 = tf.nn.elu(my_batch_norm_layer(hidden2_drop))
        else:
            bn2 = hidden2_drop
        ###Output Layer
        logits_before_bn = my_dense_layer(bn2, n_outputs, activation=None, name="outputs")
        logits = my_batch_norm_layer(logits_before_bn)
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")

    with tf.name_scope("train"):
        optimizer = tf.train.AdamOptimizer(learning_rate , name="optimizer")
        training_op = optimizer.minimize(loss)

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
    with tf.name_scope("init_and_save"):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        
    tf.add_to_collection('train_op', optimizer)
    tf.add_to_collection('cost_op', loss)
    tf.add_to_collection('input', X)
    tf.add_to_collection('target', y)
    tf.add_to_collection('pred', correct)
        
    loss_summary = tf.summary.scalar('Log Loss', loss)
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
    
    
    n_epochs = n_epochs
    batch_size = batch_size
    n_batches = len(X_train) // batch_size



    
    
       
    with tf.Session() as sess:
        init.run(session=sess)
        for epoch in range(n_epochs):
            for iteration in range(n_batches):
                
                
                X_batch, y_batch = fetch_batch(X_train, y_train,epoch, n_batches, iteration, batch_size)
                sess.run([training_op, extra_update_ops], feed_dict={is_training: True, X: X_batch, y: y_batch})
                
            summary_str = loss_summary.eval(feed_dict={is_training: False, X: X_val, y: y_val})
            file_writer.add_summary(summary_str, epoch)
            acc_train = accuracy.eval(feed_dict={is_training: False, X: X_batch, y: y_batch})
            acc_test = accuracy.eval(feed_dict={is_training: False, X: X_val, y: y_val})
            print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
            
        



        save_path = saver.save(sess, "./my_model_final.ckpt")
    
    
    
    file_writer.close()
    
    return tf.Graph()


def tensor_load(length,test, n_inputs=100, n_hidden=[500,300], n_outputs=2, momentum=0.9, learning_rate=0.1, batch_normal=1, dropout_rate=0.5, activation=tf.nn.relu, n_epochs=2, batch_size=250, prediction_length=1000):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    logdir = "{}/run-{}/".format(root_logdir, now)
    
    tf.reset_default_graph()
    NeuralNetStartTime = time.time()

    n_inputs = n_inputs 

    n_outputs = n_outputs
    learning_rate = learning_rate

    momentum = momentum

    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int64, shape=(None), name="y")
    is_training = tf.placeholder(tf.bool, shape=(), name='is_training')
    

    with tf.name_scope("dnn"):
        he_init = tf.contrib.layers.variance_scaling_initializer() #Don't Like Using Contrib

        my_batch_norm_layer = partial(
            tf.layers.batch_normalization,
            training=is_training,
            momentum=momentum)

        my_dense_layer = partial(
            tf.layers.dense,
        kernel_initializer=he_init, activation=activation)
        ###First Layer
        hidden1 = my_dense_layer(X, n_hidden[0], name="hidden1")
        hidden1_drop = tf.layers.dropout(hidden1, dropout_rate, training=is_training)
        if batch_normal==1:
            bn1 = tf.nn.elu(my_batch_norm_layer(hidden1_drop))
        else:
            bn1 = hidden1_drop

        ###Second Layer
        hidden2 = my_dense_layer(bn1, n_hidden[1], name="hidden2")
        hidden2_drop = tf.layers.dropout(hidden2, dropout_rate, training=is_training)
        if batch_normal==1:
            bn2 = tf.nn.elu(my_batch_norm_layer(hidden2_drop))
        else:
            bn2 = hidden2_drop
        ###Output Layer
        logits_before_bn = my_dense_layer(bn2, n_outputs, activation=None, name="outputs")
        logits = my_batch_norm_layer(logits_before_bn)
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")

    with tf.name_scope("train"):
        optimizer = tf.train.AdamOptimizer(learning_rate , name="optimizer")
        training_op = optimizer.minimize(loss)

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
    with tf.name_scope("init_and_save"):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        
    tf.add_to_collection('train_op', optimizer)
    tf.add_to_collection('cost_op', loss)
    tf.add_to_collection('input', X)
    tf.add_to_collection('target', y)
    tf.add_to_collection('pred', correct)
        
    loss_summary = tf.summary.scalar('Log Loss', loss)
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
    
    
    n_epochs = n_epochs
    batch_size = batch_size



    
    predictions= list()

    with tf.Session() as sess:
    #save_path = saver.save(sess, "./my_model_final.ckpt")
        saver.restore(sess, "./my_model_final.ckpt") #"my_model_final.ckpt"
        for i in range(int(np.ceil(length/prediction_length))):
            print ('Iter : %s' % i)
            X_new_scaled = test[i*prediction_length:(i+1)*prediction_length]
            Z = logits.eval(feed_dict={is_training:False ,X: X_new_scaled})
            prob = sess.run(tf.nn.softmax(Z))
            predictions.append((prob))
    

    
    
    file_writer.close()
    
    return predictions

def convert_to_output(predictions, length, to_csv=0, filename='Submission'):
    output = list()
    for i in range(len(predictions)):
        for j in range(len(predictions[i])):
            output.append(predictions[i][j][0])
        
    output = pd.DataFrame(output)
    if to_csv==1:
        submission = pd.DataFrame()
        submission['test_id'] = testDF['test_id']
        submission['is_duplicate'] = output
        #submission.head()
        submission.to_csv('../downloads/submission_NN_tf2.csv', index=False)

        return submission
    else:
        return output




def fetch_batch(data,label,epoch,n_batches,batch_index, batch_size):
    rnd.seed(epoch * n_batches + batch_index)          # not shown in the book
    indices = rnd.randint(len(data), size=batch_size)          # not shown
    X_batch = data[indices]   # not shown
    y_batch = label[indices]   # not shown
    return X_batch, y_batch


def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add() 
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = b"<stripped %d bytes>"%size
    return strip_def

def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))
        

