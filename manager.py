import joblib
import numpy as np
import pandas as pd
import datetime
from pathlib import Path
import itertools
import os


from sklearn import preprocessing
from sklearn import metrics

import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Embedding, Dense, BatchNormalization, Reshape
from keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.layers import Input, LSTM, Concatenate
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.sequence import pad_sequences
from time import perf_counter
from hyperopt import fmin as hpfmin, hp, tpe, Trials, space_eval, STATUS_OK



class Manager:

    def __init__(self, log_name, act_name, case_name, ts_name, out_name,win_size, net_out, net_embedding, delimiter, time_type, net_in, elapsed_time, remaining_time):
        self.log_name = log_name
        self.timestamp_name = ts_name
        self.case_name = case_name
        self.activity_name = act_name
        self.outcome_name = out_name
        self.win_size = win_size
        self.net_out = net_out
        self.net_embedding = net_embedding
        self.delimiter = delimiter

        self.time_type = time_type
        self.net_in = net_in
        self.remaining_time= remaining_time
        self.elapsed_time= elapsed_time

        self.act_dictionary = {}
        self.traces = [[]]
        self.traces_train = []
        self.traces_test = []
        self.w2v_embeddings = {}
        self.remainingTime_traces= [[]]
        self.elapsedTime_traces = [[]]


        self.best_score = np.inf
        self.best_model = None
        self.counter = 0

        self.outsize_act=0
        self.outsize_out=0

        self.leA= preprocessing.LabelEncoder()
        self.leO= preprocessing.LabelEncoder()

    def gen_internal_csv(self):
        """
        Genera un nuovo file csv aggiungendo, per ogni traccia del csv originale,
        un'ultima attività contenente l'outcome della traccia.
        """
        print("Processing CSV...")
        filename = str(Path('../Progetto-Tesi/data/sorted_csvs/' + self.log_name + ".csv").resolve())
        data = pd.read_csv(filename, delimiter = self.delimiter)

        data2 = data.filter([self.activity_name,self.case_name,self.timestamp_name,self.outcome_name], axis=1)
        if(self.time_type != -1):
            if(self.net_in == 1):
                data2 = data.filter([self.activity_name,self.case_name,self.timestamp_name,self.outcome_name, self.elapsed_time], axis=1)
            if(self.net_out == 3):
                data2 = data.filter([self.activity_name,self.case_name,self.timestamp_name,self.outcome_name, self.remaining_time], axis=1)
            if(self.net_in == 1 and self.net_out == 3):
                data2 = data.filter([self.activity_name,self.case_name,self.timestamp_name,self.outcome_name, self.elapsed_time, self.remaining_time], axis=1)

        data2[self.timestamp_name] = pd.to_datetime(data2[self.timestamp_name])
        data_updated = data2.copy()

        num_cases = len(data_updated[self.case_name].unique())
        print("NUMERO TRACCE = " + str(num_cases))
        num_events = data_updated.shape[0]
        print("NUMERO EVENTI = " + str(num_events))

        if(self.remaining_time == "nullRemaining"):
            data_updated[self.remaining_time] = -1
        if(self.elapsed_time == "nullElapsed"):
            data_updated[self.elapsed_time] = -1

        idx=0
        for i, row in data2.iterrows():

            if(i!=data2.shape[0]-1):

                elapsedTimeValue = -1
                #if it is set that time_type=-1 (no time info is used), the elapsed time will be set to default -1, a wrong value that will not be used though
                if(self.net_in == 1):
                    if(self.time_type==0):
                        elapsedTimeValue = row[self.elapsed_time]+1
                    elif(self.time_type==1):
                        elapsedTimeValue = row[self.elapsed_time]+(float(1)/float(86400))
                if (row[self.case_name]!=data2.at[i+1,self.case_name]):
                    idx+=1
                    line = pd.DataFrame({self.activity_name: row[self.outcome_name],
                                         self.case_name: row[self.case_name],
                                         self.timestamp_name: row[self.timestamp_name]+datetime.timedelta(seconds=1),
                                         self.outcome_name: row[self.outcome_name],
                                         self.elapsed_time: elapsedTimeValue,
                                         self.remaining_time: 0
                                         }, index=[idx])
                    data_updated = pd.concat([data_updated.iloc[:idx], line, data_updated.iloc[idx:]]).reset_index(drop=True)
            else: #last activity
                idx+=1
                elapsedTimeValue = -1
                #if it is set that time_type=-1 (no time info is used), the elapsed time will be set to default -1, a wrong value that will not be used though
                if(self.net_in == 1):
                    if(self.time_type==0):
                        elapsedTimeValue = row[self.elapsed_time]+1
                    elif(self.time_type==1):
                        elapsedTimeValue = row[self.elapsed_time]+(float(1)/float(86400))
                line = pd.DataFrame({self.activity_name: row[self.outcome_name],
                                     self.case_name: row[self.case_name],
                                     self.timestamp_name: row[self.timestamp_name]+datetime.timedelta(seconds=1),
                                     self.outcome_name: row[self.outcome_name],
                                     self.elapsed_time: elapsedTimeValue,
                                     self.remaining_time: 0
                                    }, index=[idx])
                data_updated = pd.concat([data_updated.iloc[:idx],line])
            idx+=1

        num_events = data_updated.shape[0]
        print("NUMERO EVENTI DOPO UPDATE = " + str(num_events))

        self.outsize_act = len(data_updated[self.activity_name].unique())
        self.outsize_out = len(data_updated[self.outcome_name].unique())

        filename = str(Path('../Progetto-Tesi/data/updated_csvs/'+self.log_name+"_updated.csv").resolve())
        data_updated.to_csv(filename, sep = self.delimiter)

        print("Done: updated CSV created")

    def csv_to_data(self):
        """
        Legge il file csv con le tracce aggiornate e costruisce un dizionario attività-chiave, dove la chiave è un intero
        unico con cui verranno codificate le tracce.
        Richiama la funzione di codifica delle tracce e della costruzione delle finestre.
        """

        print("Importing data from updated CSV...")
        filename = str(Path('../Progetto-Tesi/data/updated_csvs/' + self.log_name + "_updated.csv").resolve())
        data = pd.read_csv(filename, delimiter= self.delimiter)


        outcomes = data[self.outcome_name].unique()
        for i in range(0,len(outcomes)):
                self.act_dictionary[outcomes[i]] = i+1


        for i, event in data.iterrows():
            if event[self.activity_name] not in self.act_dictionary.keys():
                self.act_dictionary[event[self.activity_name]] = len(self.act_dictionary.keys()) + 1

        print(self.act_dictionary)

        self.getTraces(data)
        self.build_windows(0)

        print("Done: traces obtained")

    def getTraces(self,data):
        """
        Codifica ogni traccia in sequenze di interi (ovvero le codifiche di ogni attività)
        """
        self.traces = np.empty((len(data[self.case_name].unique()),), dtype=object)
        self.traces[...]=[[] for _ in range(len(data[self.case_name].unique()))]

        if(self.net_out == 3):
            self.remainingTime_traces = np.zeros((len(data[self.case_name].unique()),), dtype=object)
            self.remainingTime_traces[...]=[[] for _ in range(len(data[self.case_name].unique()))]
        else:
            self.remainingTime_traces = None

        if(self.net_in == 1):
            self.elapsedTime_traces = np.zeros((len(data[self.case_name].unique()),), dtype=object)
            self.elapsedTime_traces[...]=[[] for _ in range(len(data[self.case_name].unique()))]
        else:
            self.elapsedTime_traces = None

        if(self.net_embedding==0):
            outcomes = range(1, len(data[self.outcome_name].unique())+1)
        elif(self.net_embedding==1):
             name_outcomes = data[self.outcome_name].unique()
             outcomes = []
             for outcome in name_outcomes:
                outcomes.append(''.join(filter(str.isalnum, outcome)))

        traces_counter = 0
        for i, event in data.iterrows():
            if(self.net_embedding==0):
                activity_coded = self.act_dictionary[event[self.activity_name]]
            elif(self.net_embedding==1):
                activity_coded = ''.join(filter(str.isalnum,event[self.activity_name]))

            self.traces[traces_counter].append(activity_coded)

            if(self.time_type != -1):
                if(self.net_out == 3):
                    remainingTime = event[self.remaining_time]
                    self.remainingTime_traces[traces_counter].append(remainingTime)
                if(self.net_in == 1):
                    elapsedTime = event[self.elapsed_time]
                    self.elapsedTime_traces[traces_counter].append(elapsedTime)

            if(activity_coded in outcomes):
                traces_counter+=1

        if(self.net_embedding==0):
            for i in range(0,len(self.traces)):
                self.traces[i] = np.array(self.traces[i])
                if(self.time_type != -1):
                    if(self.net_out == 3):
                        self.remainingTime_traces[i] = np.array(self.remainingTime_traces[i])
                    if(self.net_in == 1):
                        self.elapsedTime_traces[i] = np.array(self.elapsedTime_traces[i])



        if(self.net_embedding==0):
            self.leA=self.leA.fit(list(self.act_dictionary.values()))
        elif(self.net_embedding==1):
            self.leA.fit(self.getAnumAct(self.act_dictionary.keys()))

        self.traces_train, self.traces_test = train_test_split(self.traces, test_size=0.2, random_state=42, shuffle=False)
        if(self.time_type != -1):
            if(self.net_out == 3):
                self.remainingTime_traces_train, self.remainingTime_traces_test = train_test_split(self.remainingTime_traces, test_size=0.2, random_state=42, shuffle=False)
            else:
                self.remainingTime_traces_train = None
                self.remainingTime_traces_test = None
            if(self.net_in == 1):
               self.elapsedTime_traces_train, self.elapsedTime_traces_test = train_test_split(self.elapsedTime_traces, test_size=0.2, random_state=42, shuffle=False)
            else:
                self.elapsedTime_traces_train = None
                self.elapsedTime_traces_test = None


    def getAnumAct(self,listAct):
        anumActList = []
        for act in listAct:
            anumActList.append(''.join(filter(str.isalnum, act)))

        return anumActList

    def getWord2VecEmbeddings(self,size):

        seed = 123
        w2v_modelX=  Word2Vec(vector_size=size, seed=seed,sg=0, min_count=1)
        w2v_modelX.build_vocab(self.traces,min_count=1)

        total_examples = w2v_modelX.corpus_count
        w2v_modelX.train(self.traces, total_examples=total_examples, epochs=200)
        w2v_modelX.save("models/w2v/generate_"+ self.log_name+ "_size" + str(size) + ".h5")
        self.save_w2v_vocab(w2v_modelX)

    def save_w2v_vocab(self,model):

        vocab = list(model.wv.index_to_key)
        for word in vocab:
            self.w2v_embeddings[word] = model.wv.get_vector(word)

    def encodePrefixes(self, size, X_vec):
        X_vec = pad_sequences(X_vec, maxlen=4, padding='pre', dtype=object, value='_PAD_')

        new = []
        for prefix in X_vec:
            list_temp_embed=[]
            for act in prefix:
                embed_vector = self.w2v_embeddings.get(act)
                if embed_vector is not None: # word is in the vocabulary learned by the w2v model
                    list_temp_embed.append(embed_vector)
                else:
                    list_temp_embed.append(np.zeros(shape=(size)))
            new.append(list_temp_embed)

        new=np.asarray(new)

        return new

    def build_windows(self, trainOrTest):

        if(trainOrTest == 0): #train
            traces = self.traces_train
            if(self.time_type != -1):
                if(self.net_out == 3):
                    remainingTime_traces = self.remainingTime_traces_train
                if(self.net_in == 1):
                    elapsedTime_traces = self.elapsedTime_traces_train
        elif(trainOrTest==1): #test
            traces = self.traces_test
            if(self.time_type != -1):
                if(self.net_out == 3):
                    remainingTime_traces = self.remainingTime_traces_test
                if(self.net_in == 1):
                    elapsedTime_traces = self.elapsedTime_traces_test

        self.act_windows = []
        self.elapsedTime_windows = []

        self.out1_labels = []
        self.out2_labels= []

        print("Building windows...")
        for i in range(0,len(traces)):
            trace = traces[i]
            if(self.time_type != -1):
                if(self.net_out == 3):
                    remainingTime_trace =  remainingTime_traces[i]
                if(self.net_in == 1):
                    elapsedTime_trace= elapsedTime_traces[i]

            k=0
            j=1
            while j < len(trace):
                if(self.net_embedding==0):
                    current_example = np.zeros(self.win_size)
                else:
                    current_example= []

                values = trace[0:j] if j <= self.win_size else \
                     trace[j - self.win_size:j]

                current_example[self.win_size - len(values):] = values
                self.act_windows.append(current_example)

                if(self.net_in == 1):
                    time_current_example = np.zeros(self.win_size)
                    time_values = elapsedTime_trace[0:j] if j <= self.win_size else \
                         elapsedTime_trace[j - self.win_size:j]
                    time_current_example[self.win_size - len(values):] = time_values
                    self.elapsedTime_windows.append(time_current_example)


                if(self.net_out == 0 or self.net_out == 1):
                    self.out1_labels.append(trace[j])
                if(self.net_out == 0 or self.net_out == 2):
                    encoded_outcome = trace[len(trace)-1]
                    self.out2_labels.append(encoded_outcome)
                if(self.net_out == 3):
                    self.out1_labels.append(remainingTime_trace[k])
                j += 1
                k+=1

        if(self.net_embedding==0):
            self.act_windows = np.asarray(self.act_windows)
            self.elapsedTime_windows = np.asarray(self.elapsedTime_windows)
            self.out1_labels = np.asarray(self.out1_labels)
            self.out2_labels= np.asarray(self.out2_labels)

        print("Done: windows built")

    def nn(self,params):
            start_time = perf_counter()
            print(start_time)

            labels = []
            if(self.net_out==0):
                NA_labels = self.leA.transform(self.out1_labels)
                NA_labels = to_categorical(NA_labels)
                labels.append(NA_labels)

                OUT_labels = self.leO.fit_transform(self.out2_labels)
                OUT_labels = to_categorical(OUT_labels)
                labels.append(OUT_labels)
            elif(self.net_out==1):
                NA_labels = self.leA.transform(self.out1_labels)
                NA_labels = to_categorical(NA_labels)
                labels = NA_labels
            elif(self.net_out==2):
                OUT_labels = self.leO.fit_transform(self.out2_labels)
                OUT_labels = to_categorical(OUT_labels)
                labels = OUT_labels
            elif(self.net_out==3):
                labels= np.asarray(self.out1_labels)


            if(self.net_embedding==0):
                act_windows = self.act_windows
                unique_events = len(self.act_dictionary)
                l_input = Input(shape=self.win_size, dtype='int32', name='input_act')
                emb_input = Embedding(output_dim=params["output_dim_embedding"], input_dim=unique_events + 1, input_length=self.win_size)(l_input)
                toBePassed=emb_input
            elif(self.net_embedding==1):
                self.getWord2VecEmbeddings(params['word2vec_size'])
                act_windows=self.encodePrefixes(params['word2vec_size'],self.act_windows)
                l_input = Input(shape = (self.win_size, params['word2vec_size']), name = 'input_act')
                if(self.net_in==1):
                    self.elapsedTime_windows = np.asarray(self.elapsedTime_windows)
                toBePassed=l_input


            if(self.net_in == 1):
                elapsed_time_input = Input(shape=self.win_size, name='input_time')
                elapsed_time_input = Reshape((-1, 1))(elapsed_time_input)
                input_concat = Concatenate(axis=-1)([toBePassed, elapsed_time_input])
                toBePassed = input_concat


            l1 = LSTM(params["shared_lstm_size"],return_sequences=True, kernel_initializer='glorot_uniform',dropout=params['dropout'])(toBePassed)
            l1 = BatchNormalization()(l1)

            n_layers = int(params["n_layers"]["n_layers"])
            if(self.net_out!=2):
                l_actORtime = LSTM(params["lstmA_size_1"], return_sequences=(n_layers != 1), kernel_initializer='glorot_uniform',dropout=params['dropout'])(l1)
                l_actORtime = BatchNormalization()(l_actORtime)
            if(self.net_out==0 or self.net_out == 2):
                l_o = LSTM(params["lstmO_size_1"],  return_sequences=(n_layers != 1), kernel_initializer='glorot_uniform',dropout=params['dropout'])(l1)
                l_o = BatchNormalization()(l_o)

            for i in range(2,n_layers+1):
                if(self.net_out!=2):
                    l_actORtime = LSTM(params["n_layers"]["lstmA_size_%s_%s" % (i, n_layers)],  return_sequences=(n_layers != i), kernel_initializer='glorot_uniform',dropout=params['dropout'])(l_actORtime)
                    l_actORtime = BatchNormalization()(l_actORtime)

                if(self.net_out==0 or self.net_out == 2):
                    l_o = LSTM(params["n_layers"]["lstmO_size_%s_%s" % (i, n_layers)],  return_sequences=(n_layers != i), kernel_initializer='glorot_uniform',dropout=params['dropout'])(l_o)
                    l_o = BatchNormalization()(l_o)

            outputLayers=[]
            if(self.net_out==0 or self.net_out==1):
                output_a = Dense(self.outsize_act, activation='softmax', name='act_output')(l_actORtime)
                outputLayers.append(output_a)
            if(self.net_out==0 or self.net_out==2):
                output_o = Dense(self.outsize_out, activation='softmax', name='outcome_output')(l_o)
                outputLayers.append(output_o)
            if(self.net_out==3):
                output_t = Dense(1, kernel_initializer='glorot_uniform', name='time_output')(l_actORtime)
                outputLayers.append(output_t)

            inputLayers = []
            inputLayers.append(l_input)
            if(self.net_in == 1):
                inputLayers.append(elapsed_time_input)

            model = Model(inputs=inputLayers, outputs=outputLayers)
            print(model.summary())

            opt = Adam(lr=params["learning_rate"])

            loss_weights= [1,1]
            metrics = 'accuracy'
            if(self.net_out==0):
                loss = {'act_output':'categorical_crossentropy', 'outcome_output':'categorical_crossentropy'}
                loss_weights= [params['gamma'], 1-params['gamma']]
            if(self.net_out==1):
                loss = {'act_output':'categorical_crossentropy'}
            if(self.net_out==2):
                loss = {'outcome_output':'categorical_crossentropy'}
            if(self.net_out==3):
                loss={'time_output':'mae'}
                metrics = None

            model.compile(loss=loss, optimizer=opt, loss_weights=loss_weights ,metrics=metrics)
            early_stopping = EarlyStopping(monitor='val_loss',
                                           patience=20)
            lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto',
                                           min_delta=0.0001, cooldown=0, min_lr=0)

            inputs = []
            inputs.append(np.asarray(act_windows))
            if(self.net_in == 1):
                inputs.append(np.asarray(self.elapsedTime_windows))

            history = model.fit(inputs, labels, epochs=1, batch_size=2**params['batch_size'], verbose=2, callbacks=[early_stopping, lr_reducer], validation_split =0.2 )

            scores = [history.history['val_loss'][epoch] for epoch in range(len(history.history['loss']))]

            score = min(scores)
            #global best_score, best_model
            if self.best_score > score:
                    self.best_score = score
                    self.best_model = model

            end_time = perf_counter()


            return {'loss': score, 'status': STATUS_OK, 'time': end_time - start_time}

    def plot_confusion_matrix(self,cm, classes,
                            normalize=False,
                            title='Confusion matrix',
                            cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        #plt.show()

    def evaluate_model(self,model,embedding_size):


        self.build_windows(1)

        if(self.net_embedding==1):
            w2v_model = Word2Vec.load("models/w2v/generate_"+ self.log_name+ "_size" + str(embedding_size) + ".h5")
            self.save_w2v_vocab(w2v_model)
            self.act_windows = self.encodePrefixes(embedding_size,self.act_windows)
            if(self.net_in==1):
                self.elapsedTime_windows = np.asarray(self.elapsedTime_windows)


        inputs = []
        inputs.append(self.act_windows)
        if(self.net_in == 1):
            inputs.append(self.elapsedTime_windows)


        if(self.net_out==0 or self.net_out ==1):
                NA_labels = self.leA.transform(self.out1_labels)
                print(NA_labels)
        if(self.net_out==0 or self.net_out == 2):
                OUT_labels = self.leO.transform(self.out2_labels)
                print(OUT_labels)
        if(self.net_out==3):
                TIME_labels = self.out1_labels
                print(TIME_labels)

        prediction = model.predict(inputs, batch_size=128, verbose = 0)
        if(self.net_out==0):
            act_pred = prediction[0]
            out_pred = prediction[1]
        elif(self.net_out==1):
            act_pred = prediction
        elif(self.net_out==2):
            out_pred = prediction
        elif(self.net_out==3):
            time_pred = prediction


        if(self.net_out==0 or self.net_out==1):
            rounded_act_prediction = np.argmax(act_pred,axis=-1)

            cm_act = confusion_matrix(y_true= NA_labels, y_pred=rounded_act_prediction)

            reportNA=metrics.classification_report(NA_labels, rounded_act_prediction, digits=3)
            print(reportNA)
            if(self.net_out==1):
                return reportNA,cm_act

        if(self.net_out==0 or self.net_out==2):
            rounded_out_prediction = np.argmax(out_pred,axis=-1)
            cm_out = confusion_matrix(y_true= OUT_labels, y_pred=rounded_out_prediction)

            reportO = metrics.classification_report(OUT_labels, rounded_out_prediction, digits=3)
            print(reportO)
            if(self.net_out==2):
                return reportO,cm_out
            elif(self.net_out==0):
                return reportNA,cm_act,reportO,cm_out

        if(self.net_out==3):
            mae = metrics.mean_absolute_error(TIME_labels, time_pred)
            return mae



    def fmin(self,
        fn,
        space,
        algo,
        max_evals,
        filename,
        rstate=None,
        pass_expr_memo_ctrl=None,
        verbose=0,
        max_queue_len=1,
        show_progressbar=True,
    ):
        """Minimize a function with hyperopt and save results to disk for later restart.
        Parameters
        ----------
        fn : callable (trial point -> loss)
            This function will be called with a value generated from `space`
            as the first and possibly only argument.  It can return either
            a scalar-valued loss, or a dictionary.  A returned dictionary must
            contain a 'status' key with a value from `STATUS_STRINGS`, must
            contain a 'loss' key if the status is `STATUS_OK`. Particular
            optimization algorithms may look for other keys as well.  An
            optional sub-dictionary associated with an 'attachments' key will
            be removed by fmin its contents will be available via
            `trials.trial_attachments`. The rest (usually all) of the returned
            dictionary will be stored and available later as some 'result'
            sub-dictionary within `trials.trials`.
        space : hyperopt.pyll.Apply node
            The set of possible arguments to `fn` is the set of objects
            that could be created with non-zero probability by drawing randomly
            from this stochastic program involving involving hp_<xxx> nodes
            (see `hyperopt.hp` and `hyperopt.pyll_utils`).
        algo : search algorithm
            This object, such as `hyperopt.rand.suggest` and
            `hyperopt.tpe.suggest` provides logic for sequential search of the
            hyperparameter space.
        max_evals : int
            Allow up to this many additional function evaluations before returning.
        filename : str, pathlib.Path, or file object
            Filename where to store the results for later restart. Results will be
            stored as a pickled hyperopt Trials object which is gziped.
        rstate : numpy.RandomState, default numpy.random or `$HYPEROPT_FMIN_SEED`
            Each call to `algo` requires a seed value, which should be different
            on each call. This object is used to draw these seeds via `randint`.
            The default rstate is
            `numpy.random.RandomState(int(env['HYPEROPT_FMIN_SEED']))`
            if the `HYPEROPT_FMIN_SEED` environment variable is set to a non-empty
            string, otherwise np.random is used in whatever state it is in.
        verbose : int
            Print out some information to stdout during search.
        pass_expr_memo_ctrl : bool, default False
            If set to True, `fn` will be called in a different more low-level
            way: it will receive raw hyperparameters, a partially-populated
            `memo`, and a Ctrl object for communication with this Trials
            object.
        max_queue_len : integer, default 1
            Sets the queue length generated in the dictionary or trials. Increasing this
            value helps to slightly speed up parallel simulatulations which sometimes lag
            on suggesting a new trial.
        show_progressbar : bool, default True
            Show a progressbar.
        Returns
        -------
        list : (argmin, trials)
            argmin : dictionary
                If return_argmin is True returns `trials.argmin` which is a dictionary.  Otherwise
                this function  returns the result of `hyperopt.space_eval(space, trails.argmin)` if there
                were succesfull trails. This object shares the same structure as the space passed.
                If there were no succesfull trails, it returns None.
            trials : hyperopt.Trials
                The hyperopt trials object that also gets stored to disk.
        """
        new_max_evals = max_evals
        newFilename = '../Progetto-Tesi/models/hpTrials/' + filename + '_Trials'

        try:
            trials = joblib.load(newFilename)
            evals_loaded_trials = len(trials.statuses())
            new_max_evals = max_evals+ evals_loaded_trials
            print('{} evals loaded from trials file "{}".'.format(evals_loaded_trials, filename))
        except FileNotFoundError:
            trials = Trials()
            print('No trials file "{}" found. Created new trials object.'.format(filename))


        result = hpfmin(
            fn,
            space,
            algo,
            new_max_evals,
            trials=trials,
            rstate=rstate,
            pass_expr_memo_ctrl=pass_expr_memo_ctrl,
            verbose=verbose,
            return_argmin=True,
            max_queue_len=max_queue_len,
            show_progressbar=show_progressbar,
        )

        joblib.dump(trials, newFilename, compress=("gzip", 3))

        bestTrial = trials.best_trial
        print("The best score for the last run is : ", self.best_score)
        if(self.best_score>float(bestTrial['result']['loss'])):
            print("New best score/model found")
            print(float(bestTrial['result']['loss']))
            self.best_model = load_model("models/generate_"+filename+".h5")

        return result, trials


    pass
