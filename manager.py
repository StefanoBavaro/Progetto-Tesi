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
from keras.layers import Input, LSTM
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.sequence import pad_sequences
from time import perf_counter
from hyperopt import fmin as hpfmin, hp, tpe, Trials, space_eval, STATUS_OK



class Manager:

    def __init__(self, log_name, act_name, case_name, ts_name, out_name,win_size, net_out, net_embedding, delimiter, time_type = -1):
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

        if(self.time_type == "seconds"):
            self.ending_name = "seconds_to_end"
            self.elapsed_time = "elapsed_seconds"
        elif(self.time_type=="days"):
            self.ending_name = "days_to_end"
            self.elapsed_time = "elapsed_days"
        #self.counter=0

        self.act_dictionary = {}
        self.traces = [[]]
        self.traces_train = []
        self.traces_test = []
        #self.w2v_act = []
        self.w2v_embeddings = {}
        #self.word_vec_dict = {}
        #self.example_size = ex_size
        self.seconds_traces= [[]]


        # self.x_training=[]
        # self.y_training=[]
        # self.z_training=[]

        self.best_score = np.inf
        self.best_model = None

       # self.words_emb = []

        self.outsize_act=0
        self.outsize_out=0

        # self.leA= preprocessing.OrdinalEncoder(handle_unknown='use_encoded_value',
        #                          unknown_value=-1)
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
        data2[self.timestamp_name] = pd.to_datetime(data2[self.timestamp_name])
        data_updated = data2.copy()

        num_cases = len(data_updated[self.case_name].unique())
        print("NUMERO TRACCE = " + str(num_cases))
        num_events = data_updated.shape[0]
        print("NUMERO EVENTI = " + str(num_events))

        #print(data2.shape[0])
        idx=0
        for i, row in data2.iterrows():
            # print(i)
            # print(idx)
            if(i!=data2.shape[0]-1):
                if (row[self.case_name]!=data2.at[i+1,self.case_name]):
                    idx+=1
                    line = pd.DataFrame({self.activity_name: row[self.outcome_name],
                                         self.case_name: row[self.case_name],
                                         self.timestamp_name: row[self.timestamp_name]+datetime.timedelta(seconds=1),
                                         self.outcome_name: row[self.outcome_name]
                                         }, index=[idx])
                    data_updated = pd.concat([data_updated.iloc[:idx], line, data_updated.iloc[idx:]]).reset_index(drop=True)
            else: #ultima attività(ultimo caso), serve aggiungere l'evento outcome comunque
                #print("last")
                idx+=1
                line = pd.DataFrame({self.activity_name: row[self.outcome_name],
                                               self.case_name: row[self.case_name],
                                               self.timestamp_name: row[self.timestamp_name]+datetime.timedelta(seconds=1),
                                               self.outcome_name: row[self.outcome_name]
                                               }, index=[idx])
                data_updated = pd.concat([data_updated.iloc[:idx],line])
            idx+=1

        num_events = data_updated.shape[0]
        print("NUMERO EVENTI DOPO UPDATE = " + str(num_events))

        self.outsize_act = len(data_updated[self.activity_name].unique())
        #print(self.outsize_act)
        self.outsize_out = len(data_updated[self.outcome_name].unique())
        #print(self.outsize_out)

        filename = str(Path('../Progetto-Tesi/data/updated_csvs/'+self.log_name+"_updated.csv").resolve())
        data_updated.to_csv(filename, sep = self.delimiter)
        #print(data_updated)
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

        #inserisco per prime nel dizionario le label degli outcome
        outcomes = data[self.outcome_name].unique()
        for i in range(0,len(outcomes)):
            if(self.net_embedding==0):
                self.act_dictionary[outcomes[i]] = i+1
            # if(self.net_embedding==1):
            #     self.w2v_act.append(''.join(filter(str.isalnum, outcomes[i])))

        #inserisco il resto delle attività nel dizionario
        for i, event in data.iterrows():
            if event[self.activity_name] not in self.act_dictionary.keys():
                self.act_dictionary[event[self.activity_name]] = len(self.act_dictionary.keys()) + 1
                # if(self.net_embedding==1):
                #     # if('' not in self.w2v_act):
                #     #     self.w2v_act.append('')
                #     self.w2v_act.append(''.join(filter(str.isalnum, event[self.activity_name])))

        print(self.act_dictionary)
        #print(self.w2v_act)

        #if(self.net_embedding==0):
        self.getTraces(data)

        print("Done: traces obtained")
        # else:
        #     self.getWord2VecEmbeddings(data)

    def getTraces(self,data):
        """
        Codifica ogni traccia in sequenze di interi (ovvero le codifiche di ogni attività)
        """
        self.traces = np.empty((len(data[self.case_name].unique()),), dtype=object)
        self.traces[...]=[[] for _ in range(len(data[self.case_name].unique()))]
        if(self.net_embedding==0):
            outcomes = range(1, len(data[self.outcome_name].unique())+1)
        elif(self.net_embedding==1):
             name_outcomes = data[self.outcome_name].unique()
             outcomes = []
             for outcome in name_outcomes:
                outcomes.append(''.join(filter(str.isalnum, outcome)))
        #print(outcomes)

        traces_counter = 0
        for i, event in data.iterrows():
            if(self.net_embedding==0):
                activity_coded = self.act_dictionary[event[self.activity_name]]
            elif(self.net_embedding==1):
                activity_coded = ''.join(filter(str.isalnum,event[self.activity_name]))
                #activity_coded = ''.join(filter(str.isalnum, activity_coded))
            #print(activity_coded)
            self.traces[traces_counter].append(activity_coded)
            #print(self.traces[traces_counter])
            if(activity_coded in outcomes):
                traces_counter+=1

        #converto ogni traccia in array piuttosto che liste
        if(self.net_embedding==0):
            for i in range(0,len(self.traces)):
                self.traces[i] = np.array(self.traces[i])

        #print(list(self.act_dictionary.values()))
        if(self.net_embedding==0):
            self.leA=self.leA.fit(list(self.act_dictionary.values()))
        elif(self.net_embedding==1):
            self.leA.fit(self.getAnumAct(self.act_dictionary.keys()))
        #print(self.traces)
        self.traces_train, self.traces_test = train_test_split(self.traces, test_size=0.2, random_state=42, shuffle=False)
        # X_train,Y_train,Z_train = self.build_windows(self.traces_train,self.win_size)
        # print(X_train)
        # print(Y_train)
        # print(Z_train)

        # print("train")
        # print(self.traces_train)
        #
        # print("test")
        # print(self.traces_test)


    def getAnumAct(self,listAct):

        anumActList = []
        for act in listAct:
            anumActList.append(''.join(filter(str.isalnum, act)))

        return anumActList

    def getWord2VecEmbeddings(self,size):
        #print(size)
        #print(len(X_train))
        #print(len(self.w2v_act))
        # print("NUM TRACCE")
        # print(len(self.traces))

        seed = 123
        w2v_modelX=  Word2Vec(vector_size=size, seed=seed,sg=0, min_count=1)
        #print(self.w2v_act)
        w2v_modelX.build_vocab(self.traces,min_count=1)

        total_examples = w2v_modelX.corpus_count
        # print("NUM corpus w2v")
        # print(total_examples)
        # addestro W2V
        w2v_modelX.train(self.traces, total_examples=total_examples, epochs=200)
        # salvo modello W2V
        w2v_modelX.save("models/w2v/generate_"+ self.log_name+ "_size" + str(size) + ".h5")
        self.save_w2v_vocab(w2v_modelX)

    def save_w2v_vocab(self,model):

        #print('new W2V models-->',"w2v/" + self.log_name + "/generate_" + self.log_name + '_' + str(self.count_model) + ".h5")
        vocab = list(model.wv.index_to_key)
        # print(vocab)

        #w2vModelX.train(X_train, total_examples=len([self.w2v_act]), epochs=10)

        # print("The total number of words are : ", len(vocab))
        # print("The num of words :", len(vocab))
        # print("The words in vocab :", (vocab))

        for word in vocab:
            self.w2v_embeddings[word] = model.wv.get_vector(word)

    def encodePrefixes(self, size, X_vec):
        #print('PRE',X_vec)
        X_vec = pad_sequences(X_vec, maxlen=4, padding='pre', dtype=object, value='_PAD_')
        #print('PRE',X_vec)
        # self.list_emb = []
        # for l in pad_rev:
        #   list_emb_temp = []
        #   for t in l:
        #       embed_vector = self.word_vec_dict.get(t)
        #       if embed_vector is not None: # word is in the vocabulary learned by the w2v model
        #           list_emb_temp.append(embed_vector)
        #       else:
        #           list_emb_temp.append(np.zeros(shape=(self.vec_dim)))
        # self.list_emb.append(list_emb_temp)
        # self.list_emb = np.asarray(self.list_emb)


        new = []
        for prefix in X_vec:
            list_temp_embed=[]
            #i = 0
            for act in prefix:
                embed_vector = self.w2v_embeddings.get(act)
                if embed_vector is not None: # word is in the vocabulary learned by the w2v model
                    list_temp_embed.append(embed_vector)
                    #print(len(prefix[i]))
                else:
                    list_temp_embed.append(np.zeros(shape=(size)))
                    #print(len(prefix[i]))
            new.append(list_temp_embed)
                #print(len(prefix))
                #i+=1
        new=np.asarray(new)

        # print("SHAPE:",np.shape(X_vec))
        # X_vec= np.array([np.array(val) for val in X_vec])
        # print("SHAPE:",np.shape(X_vec))
        #print('POST',X_vec)
        # print(X_vec)

        #print(np.shape(X_train))
        #print("END WORD2VEC FUNCTION")
        return new

    def build_windows(self,traces,win_size):
        X_vec = []
        Y_vec = []
        Z_vec = []

        print("Building windows...")
        for trace in traces:
            # print("TRACE:")
            # print(trace)
            i=0
            #print(i)
            j=1
            #while j < trace.size:
            while j < len(trace):
                #print(j)
                if(self.net_embedding==0):
                    current_example = np.zeros(win_size)
                else:
                    current_example= []
                values = trace[0:j] if j <= win_size else \
                         trace[j - win_size:j]
                Y_vec.append(trace[j])
                #print(values)
                current_example[win_size - len(values):] = values
                #print(current_example)
                X_vec.append(current_example)
                encoded_outcome = trace[len(trace)-1]
                Z_vec.append(encoded_outcome)
                j += 1
            i+=1

        if(self.net_embedding==0):
            X_vec = np.asarray(X_vec)
            Y_vec = np.asarray(Y_vec)
            Z_vec = np.asarray(Z_vec)




        print("Done: windows built")
        return X_vec,Y_vec,Z_vec

    def nn(self,params):
            #it is not done before so that, in case, win_size can become a parameter
            start_time = perf_counter()
            print(start_time)

            X_train,Y_train,Z_train = self.build_windows(self.traces_train,self.win_size)

            # print(X_train);
            # print(Y_train);
            # print(Z_train);

            #label=None

            if(self.net_out!=2):
                #self.outsize_act = len(np.unique(Y_train))
                #print(np.unique(Y_train))
                #Y_train = self.leA.fit_transform(Y_train.reshape(-1, 1))
                Y_train = self.leA.transform(Y_train)
                #print(np.unique(Y_train))
                #print(Y_train)
                Y_train = to_categorical(Y_train)
                #print(Y_train)

                label=Y_train
            if(self.net_out!=1):
                Z_train = self.leO.fit_transform(Z_train)
                #print(Z_train)
                Z_train = to_categorical(Z_train)
                #print(Z_train)
                label=Z_train

            unique_events = len(self.act_dictionary) #numero di diversi eventi/attività nel dataset
            #print(unique_events)
            #size_act = (unique_events + 1) // 2

            n_layers = int(params["n_layers"]["n_layers"])

            if(self.net_embedding==0):
                #l_input = Input(shape=self.win_size, dtype='int32', name='input_act')
                l_input = Input(shape=self.win_size, dtype='int32', name='input_act')
                emb_input = Embedding(output_dim=params["output_dim_embedding"], input_dim=unique_events + 1, input_length=self.win_size)(l_input)
                toBePassed=emb_input
            elif(self.net_embedding==1):
                self.getWord2VecEmbeddings(params['word2vec_size'])
                X_train=self.encodePrefixes(params['word2vec_size'],X_train)
                # print(type(X_train))
                # print(np.shape(X_train))
                l_input = Input(shape = (self.win_size, params['word2vec_size']), name = 'input_act')
                toBePassed=l_input



            l1 = LSTM(params["shared_lstm_size"],return_sequences=True, kernel_initializer='glorot_uniform',dropout=params['dropout'])(toBePassed)
            l1 = BatchNormalization()(l1)
            if(self.net_out!=2):
                l_a = LSTM(params["lstmA_size_1"], return_sequences=(n_layers != 1), kernel_initializer='glorot_uniform',dropout=params['dropout'])(l1)
                l_a = BatchNormalization()(l_a)
            if(self.net_out!=1):
                l_o = LSTM(params["lstmO_size_1"],  return_sequences=(n_layers != 1), kernel_initializer='glorot_uniform',dropout=params['dropout'])(l1)
                l_o = BatchNormalization()(l_o)

            for i in range(2,n_layers+1):
                if(self.net_out!=2):
                    l_a = LSTM(params["n_layers"]["lstmA_size_%s_%s" % (i, n_layers)],  return_sequences=(n_layers != i), kernel_initializer='glorot_uniform',dropout=params['dropout'])(l_a)
                    l_a = BatchNormalization()(l_a)

                if(self.net_out!=1):
                    l_o = LSTM(params["n_layers"]["lstmO_size_%s_%s" % (i, n_layers)],  return_sequences=(n_layers != i), kernel_initializer='glorot_uniform',dropout=params['dropout'])(l_o)
                    l_o = BatchNormalization()(l_o)

            outputs=[]
            if(self.net_out!=2):
                output_l = Dense(self.outsize_act, activation='softmax', name='act_output')(l_a)
                outputs.append(output_l)
            if(self.net_out!=1):
                output_o = Dense(self.outsize_out, activation='softmax', name='outcome_output')(l_o)
                outputs.append(output_o)

            model = Model(inputs=l_input, outputs=outputs)
            print(model.summary())

            opt = Adam(lr=params["learning_rate"])

            if(self.net_out==0):
                loss = {'act_output':'categorical_crossentropy', 'outcome_output':'categorical_crossentropy'}
                loss_weights= [params['gamma'], 1-params['gamma']]
            if(self.net_out==1):
                loss = {'act_output':'categorical_crossentropy'}
                loss_weights= [1,1]
            if(self.net_out==2):
                loss = {'outcome_output':'categorical_crossentropy'}
                loss_weights=[1,1]

            model.compile(loss=loss, optimizer=opt, loss_weights=loss_weights ,metrics=['accuracy'])
            early_stopping = EarlyStopping(monitor='val_loss',
                                           patience=20)
            lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto',
                                           min_delta=0.0001, cooldown=0, min_lr=0)

            if(self.net_out==0):
                history = model.fit(X_train, [Y_train,Z_train], epochs=300, batch_size=2**params['batch_size'], verbose=2, callbacks=[early_stopping, lr_reducer], validation_split =0.2 )
                # if(self.net_embedding==0):
                #     history = model.fit(X_train, [Y_train,Z_train], epochs=3, batch_size=2**params['batch_size'], verbose=2, callbacks=[early_stopping, lr_reducer], validation_split =0.2 )
                # elif(self.net_embedding==1):
                #     #X_train, X_val, Y_train,Y_val,Z_train,Z_val= train_test_split(X_train,Y_train,Z_train, test_size=0.2, random_state=42, shuffle=False)
                #     #history = models.fit(X_train, [Y_train,Z_train], epochs=3, batch_size=2**params['batch_size'], verbose=2, callbacks=[early_stopping, lr_reducer], validation_data = (X_val, [Y_val, Z_val]))
                #     history = model.fit(np.asarray(X_train), [np.asarray(Y_train),np.asarray(Z_train)], epochs=3, batch_size=2**params['batch_size'], verbose=2, callbacks=[early_stopping, lr_reducer], validation_split =0.2 )
            else:
                history = model.fit(X_train, label, epochs=300, batch_size=2**params['batch_size'], verbose=2, callbacks=[early_stopping, lr_reducer], validation_split =0.2 )


            scores = [history.history['val_loss'][epoch] for epoch in range(len(history.history['loss']))]



            score = min(scores)
            #global best_score, best_model
            if self.best_score > score:
                    self.best_score = score
                    self.best_model = model

            end_time = perf_counter()


            return {'loss': score, 'status': STATUS_OK, 'time': end_time - start_time}
            #models.save("models/generate_" + self.log_name + ".h5")

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
        #models = load_model("models/generate_" + self.log_name + ".h5")

        X_test, Y_test, Z_test = self.build_windows(self.traces_test,self.win_size)

        if(self.net_out!=2):
            print(np.unique(Y_test))
            #Y_test = self.leA.transform(Y_test.reshape(-1,1))
            Y_test= self.leA.transform(Y_test)
            print(np.unique(Y_test))
            #print(Y_test)
            #print(np.unique(Y_test))
        if(self.net_out!=1):
            Z_test = self.leO.transform(Z_test)
        #print(Y_test)

        if(self.net_embedding==1):
             #w2v_modelX.save("models/w2v/generate_"+ self.log_name+ "_size" + str(size) + ".h5")

            w2v_model = Word2Vec.load("models/w2v/generate_"+ self.log_name+ "_size" + str(embedding_size) + ".h5")
            self.save_w2v_vocab(w2v_model)
            X_test = self.encodePrefixes(embedding_size,X_test)
            # X_test = np.asarray()


        prediction = model.predict(X_test, batch_size=128, verbose = 0)
        if(self.net_out==0):
            y_pred = prediction[0]
            z_pred = prediction[1]
        elif(self.net_out==1):
            y_pred = prediction
        elif(self.net_out==2):
            z_pred = prediction


        if(self.net_out!=2):
            rounded_act_prediction = np.argmax(y_pred,axis=-1)

            # print(rounded_act_prediction)
            # print(np.unique(rounded_act_prediction))
            #print(rounded_act_prediction)
            cm_act = confusion_matrix(y_true= Y_test, y_pred=rounded_act_prediction)
            cm_plot_labels_act = range(0,self.outsize_act)
            #print(cm_act)
            self.plot_confusion_matrix(cm=cm_act, classes=cm_plot_labels_act, title='Confusion Matrix Next Activity')
            reportNA=metrics.classification_report(Y_test, rounded_act_prediction, digits=3)
            print(reportNA)
            if(self.net_out==1):
                return reportNA,cm_act

        if(self.net_out!=1):
            rounded_out_prediction = np.argmax(z_pred,axis=-1)
            cm_out = confusion_matrix(y_true= Z_test, y_pred=rounded_out_prediction)
            cm_plot_labels_out = range(0,self.outsize_out)
            #print(cm_out)
            self.plot_confusion_matrix(cm=cm_out, classes=cm_plot_labels_out, title='Confusion Matrix Outcome')
            reportO = metrics.classification_report(Z_test, rounded_out_prediction, digits=3)
            print(reportO)
            if(self.net_out==2):
                return reportO,cm_out

        return reportNA,cm_act,reportO,cm_out


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
        try:
            trials = joblib.load(filename+'_Trials')
            evals_loaded_trials = len(trials.statuses())
            new_max_evals = max_evals+ evals_loaded_trials
            print('{} evals loaded from trials file "{}".'.format(evals_loaded_trials, filename))
        except FileNotFoundError:
            trials = Trials()
            print('No trials file "{}" found. Created new trials object.'.format(filename))

        if(self.net_out!=3):
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
        else:
            result = hpfmin(
                self.timePred_nn,
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

        joblib.dump(trials, filename+'_Trials', compress=("gzip", 3))

        bestTrial = trials.best_trial
        print("The best score for the last run is : ", self.best_score)
        if(self.best_score>float(bestTrial['result']['loss'])):
            print("New best score/model found")
            print(float(bestTrial['result']['loss']))
            self.best_model = load_model("models/generate_" + self.log_name + "_type"+str(self.net_out)+"_emb"+str(self.net_embedding) + ".h5")

        return result, trials


    def gen_internal_csv_timeNet(self):
        """
        Genera un nuovo file csv aggiungendo, per ogni traccia del csv originale,
        un'ultima attività contenente l'outcome della traccia.
        """

        ##In questa tipologia di dataset il remaining_time (days or seconds to end) è già incrementato di uno (per l'aggiunta di una nuova attivita)
        print("Processing CSV...")
        filename = str(Path('../Progetto-Tesi/data/sorted_csvs/' + self.log_name + ".csv").resolve())
        data = pd.read_csv(filename, delimiter = self.delimiter)


        data2 = data.filter([self.activity_name,self.case_name,self.timestamp_name,self.outcome_name,self.ending_name, self.elapsed_time], axis=1)
        data2[self.timestamp_name] = pd.to_datetime(data2[self.timestamp_name])
        data2[self.ending_name]
        data_updated = data2.copy()

        num_cases = len(data_updated[self.case_name].unique())
        print("NUMERO TRACCE = " + str(num_cases))
        num_events = data_updated.shape[0]
        print("NUMERO EVENTI = " + str(num_events))

        #print(data2.shape[0])
        idx=0
        for i, row in data2.iterrows():
            # print(i)
            # print(idx)
            if(i!=data2.shape[0]-1):
                if (row[self.case_name]!=data2.at[i+1,self.case_name]):
                    idx+=1
                    elapsedTimeValue = 0
                    if(self.time_type=="seconds"):
                        elapsedTimeValue = row[self.elapsed_time]+1
                    elif(self.time_type =="days"):
                        elapsedTimeValue = float(row["elapsed_seconds"]+1) /float(86400)

                    line = pd.DataFrame({self.activity_name: row[self.outcome_name],
                                         self.case_name: row[self.case_name],
                                         self.timestamp_name: row[self.timestamp_name]+datetime.timedelta(seconds=1),
                                         self.outcome_name: row[self.outcome_name],
                                         self.elapsed_time: elapsedTimeValue,
                                         self.ending_name: 0
                                         }, index=[idx])
                    data_updated = pd.concat([data_updated.iloc[:idx], line, data_updated.iloc[idx:]]).reset_index(drop=True)
            else: #ultima attività(ultimo caso), serve aggiungere l'evento outcome comunque
                #print("last")
                idx+=1
                elapsedTimeValue = 0
                if(self.time_type=="seconds"):
                    elapsedTimeValue = row[self.elapsed_time]+1
                elif(self.time_type =="days"):
                    elapsedTimeValue = float(row["elapsed_seconds"]+1) /float(86400)
                line = pd.DataFrame({self.activity_name: row[self.outcome_name],
                                     self.case_name: row[self.case_name],
                                     self.timestamp_name: row[self.timestamp_name]+datetime.timedelta(seconds=1),
                                     self.outcome_name: row[self.outcome_name],
                                     self.elapsed_time: elapsedTimeValue,
                                     self.ending_name: 0
                                     }, index=[idx])
                data_updated = pd.concat([data_updated.iloc[:idx],line])
            idx+=1

        num_events = data_updated.shape[0]
        print("NUMERO EVENTI DOPO UPDATE = " + str(num_events))

        self.outsize_act = len(data[self.activity_name].unique())
        #print(self.outsize_act)
        self.outsize_out = len(data[self.outcome_name].unique())
        #print(self.outsize_out)



        filename = str(Path('../Progetto-Tesi/data/updated_csvs/'+self.log_name+"_updated_timeNet.csv").resolve())
        data_updated.to_csv(filename, sep = self.delimiter)
        #print(data_updated)
        print("Done: updated CSV created")

    def csv_to_data_timeNet(self):
        """
        Legge il file csv con le tracce aggiornate e costruisce un dizionario attività-chiave, dove la chiave è un intero
        unico con cui verranno codificate le tracce.
        Richiama la funzione di codifica delle tracce e della costruzione delle finestre.
        """


        print("Importing data from updated CSV...")
        filename = str(Path('../Progetto-Tesi/data/updated_csvs/' + self.log_name + "_updated_timeNet.csv").resolve())
        data = pd.read_csv(filename, delimiter= self.delimiter)


        #inserisco per prime nel dizionario le label degli outcome
        outcomes = data[self.outcome_name].unique()
        for i in range(0,len(outcomes)):
            self.act_dictionary[outcomes[i]] = i+1
            # if(self.net_embedding==1):
            #     self.w2v_act.append(''.join(filter(str.isalnum, outcomes[i])))

        #inserisco il resto delle attività nel dizionario
        for i, event in data.iterrows():
            if event[self.activity_name] not in self.act_dictionary.keys():
                self.act_dictionary[event[self.activity_name]] = len(self.act_dictionary.keys()) + 1
                # if(self.net_embedding==1):
                #     # if('' not in self.w2v_act):
                #     #     self.w2v_act.append('')
                #     self.w2v_act.append(''.join(filter(str.isalnum, event[self.activity_name])))

        print(self.act_dictionary)
        #print(self.w2v_act)

        #if(self.net_embedding==0):
        self.getTraces_timeNet(data)

        print("Done: traces obtained")
        # else:
        #     self.getWord2VecEmbeddings(data)

    def getTraces_timeNet(self,data):
        """
        Codifica ogni traccia in sequenze di interi (ovvero le codifiche di ogni attività)
        """
        self.traces = np.empty((len(data[self.case_name].unique()),), dtype=object)
        self.traces[...]=[[] for _ in range(len(data[self.case_name].unique()))]

        self.seconds_traces = np.zeros((len(data[self.case_name].unique()),), dtype=object)
        self.seconds_traces[...]=[[] for _ in range(len(data[self.case_name].unique()))]


        if(self.net_embedding==0):
            outcomes = range(1, len(data[self.outcome_name].unique())+1)
        elif(self.net_embedding==1):
             name_outcomes = data[self.outcome_name].unique()
             outcomes = []
             for outcome in name_outcomes:
                outcomes.append(''.join(filter(str.isalnum, outcome)))
        #print(outcomes)

        traces_counter = 0
        for i, event in data.iterrows():
            #print("entro")
            if(self.net_embedding==0):
                activity_coded = self.act_dictionary[event[self.activity_name]]
            elif(self.net_embedding==1):
                activity_coded = ''.join(filter(str.isalnum,event[self.activity_name]))
                #activity_coded = ''.join(filter(str.isalnum, activity_coded))
            #print(activity_coded)
            seconds_to_ending = event[self.ending_name]
            print(seconds_to_ending)
            self.seconds_traces[traces_counter].append(seconds_to_ending)
            self.traces[traces_counter].append(activity_coded)
            #print(self.traces[traces_counter])
            if(activity_coded in outcomes):
                traces_counter+=1

        #converto ogni traccia in array piuttosto che liste
        if(self.net_embedding==0):
            for i in range(0,len(self.traces)):
                self.seconds_traces[i]= np.array(self.seconds_traces[i])
                self.traces[i] = np.array(self.traces[i])


        #print(list(self.act_dictionary.values()))
        self.leA=self.leA.fit(list(self.act_dictionary.values()))
        #print(self.traces)
        self.traces_train, self.traces_test = train_test_split(self.traces, test_size=0.2, random_state=42, shuffle=False)
        self.seconds_traces_train, self.seconds_traces_test = train_test_split(self.seconds_traces, test_size=0.2, random_state=42, shuffle=False)


        # X_train,Y_train,Z_train = self.build_windows(self.traces_train,self.win_size)
        # print(X_train)
        # print(Y_train)
        # print(Z_train)

        # print("train")
        # print(self.traces_train)
        #
        # print("test")
        # print(self.traces_test)

    def build_windows_timeNet(self,traces,seconds_traces,win_size):
        X_vec = [] #prefixes
        el_time_vec = []
        Y_vec = [] #

        print("Building windows...")
        for i in range(0,len(traces)):
            trace= traces[i]
            sec_trace= seconds_traces[i]
            #print(sec_trace)
            # print("TRACE:")
            # print(trace)
            k=0
            #print(i)
            j=1
            #while j < trace.size:
            while j < len(trace):
                #print(j)
                if(self.net_embedding==0):
                    current_example = np.zeros(win_size)
                else:
                    current_example= []
                time_current_example = np.zeros(win_size)

                values = trace[0:j] if j <= win_size else \
                         trace[j - win_size:j]
                time_values = sec_trace[0:j] if j <= win_size else \
                         trace[j - win_size:j]

                Y_vec.append(sec_trace[k])
                #print(values)
                current_example[win_size - len(values):] = values
                time_current_example[win_size - len(values):] = time_values
                #print(current_example)
                X_vec.append(current_example)
                el_time_vec.append(time_current_example)
                j += 1
                k+=1

        if(self.net_embedding==0):
            X_vec = np.asarray(X_vec)
            el_time_vec = np.asarray(el_time_vec)
            Y_vec = np.asarray(Y_vec)




        print("Done: windows built")
        return X_vec,el_time_vec,Y_vec




    def timePred_nn(self,params):
        #it is not done before so that, in case, win_size can become a parameter
        start_time = perf_counter()
        print(start_time)

        X_train,X2_train, t_vec = self.build_windows_timeNet(self.traces_train,self.seconds_traces_train,self.win_size)
        print(X2_train)
        #print(t_vec)
        #
        # if(self.net_out==3):
        #     t_vec=t_vec

        unique_events = len(self.act_dictionary) #numero di diversi eventi/attività nel dataset
        #print(unique_events)
        #size_act = (unique_events + 1) // 2

        n_layers = int(params["n_layers"]["n_layers"])

        if(self.net_embedding==0):
            #l_input = Input(shape=self.win_size, dtype='int32', name='input_act')
            l_input = Input(shape=self.win_size, dtype='int32', name='input_act')
            emb_input = Embedding(output_dim=params["output_dim_embedding"], input_dim=unique_events + 1, input_length=self.win_size)(l_input)
            toBePassed=emb_input
        elif(self.net_embedding==1):
            self.getWord2VecEmbeddings(params['word2vec_size'])
            X_train=self.encodePrefixes(params['word2vec_size'],X_train)
            # print(type(X_train))
            # print(np.shape(X_train))
            l_input = Input(shape = (self.win_size, params['word2vec_size']), name = 'input_act')
            toBePassed=l_input



        l1 = LSTM(params["shared_lstm_size"],return_sequences=True, kernel_initializer='glorot_uniform',dropout=params['dropout'])(toBePassed)
        l1 = BatchNormalization()(l1)
        # if(self.net_out!=2):
        l_a = LSTM(params["lstmA_size_1"], return_sequences=(n_layers != 1), kernel_initializer='glorot_uniform',dropout=params['dropout'])(l1)
        l_a = BatchNormalization()(l_a)
        # if(self.net_out!=1):
        #     l_o = LSTM(params["lstmO_size_1"],  return_sequences=(n_layers != 1), kernel_initializer='glorot_uniform',dropout=params['dropout'])(l1)
        #     l_o = BatchNormalization()(l_o)

        for i in range(2,n_layers+1):
            # if(self.net_out!=2):
            l_a = LSTM(params["n_layers"]["lstmA_size_%s_%s" % (i, n_layers)],  return_sequences=(n_layers != i), kernel_initializer='glorot_uniform',dropout=params['dropout'])(l_a)
            l_a = BatchNormalization()(l_a)

            # if(self.net_out!=1):
            #     l_o = LSTM(params["n_layers"]["lstmO_size_%s_%s" % (i, n_layers)],  return_sequences=(n_layers != i), kernel_initializer='glorot_uniform',dropout=params['dropout'])(l_o)
            #     l_o = BatchNormalization()(l_o)

        outputs=[]
        # if(self.net_out!=2):
        #     output_l = Dense(self.outsize_act, activation='softmax', name='act_output')(l_a)
        #     outputs.append(output_l)
        # if(self.net_out!=1):
        output_o = Dense(1, kernel_initializer='glorot_uniform', name='time_output')(l_a)
        outputs.append(output_o)

        model = Model(inputs=l_input, outputs=outputs)
        print(model.summary())

        opt = Adam(lr=params["learning_rate"])

        # if(self.net_out==0):
        #     loss = {'act_output':'categorical_crossentropy', 'outcome_output':'categorical_crossentropy'}
        #     loss_weights= [params['gamma'], 1-params['gamma']]
        # if(self.net_out==1):
        #     loss = {'act_output':'categorical_crossentropy'}
        #     loss_weights= [1,1]
        # if(self.net_out==2):
        #     loss = {'outcome_output':'categorical_crossentropy'}
        #     loss_weights=[1,1]

        loss={'time_output':'mae'}
        model.compile(loss=loss, optimizer=opt)
        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=20)
        lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto',
                                       min_delta=0.0001, cooldown=0, min_lr=0)

        # if(self.net_out==0):
        #     history = model.fit(X_train, [Y_train,Z_train], epochs=300, batch_size=2**params['batch_size'], verbose=2, callbacks=[early_stopping, lr_reducer], validation_split =0.2 )
        #     # if(self.net_embedding==0):
        #     #     history = model.fit(X_train, [Y_train,Z_train], epochs=3, batch_size=2**params['batch_size'], verbose=2, callbacks=[early_stopping, lr_reducer], validation_split =0.2 )
        #     # elif(self.net_embedding==1):
        #     #     #X_train, X_val, Y_train,Y_val,Z_train,Z_val= train_test_split(X_train,Y_train,Z_train, test_size=0.2, random_state=42, shuffle=False)
        #     #     #history = models.fit(X_train, [Y_train,Z_train], epochs=3, batch_size=2**params['batch_size'], verbose=2, callbacks=[early_stopping, lr_reducer], validation_data = (X_val, [Y_val, Z_val]))
        #     #     history = model.fit(np.asarray(X_train), [np.asarray(Y_train),np.asarray(Z_train)], epochs=3, batch_size=2**params['batch_size'], verbose=2, callbacks=[early_stopping, lr_reducer], validation_split =0.2 )
        # else:
        history = model.fit(X_train, np.asarray(t_vec), epochs=300, batch_size=2**params['batch_size'], verbose=2, callbacks=[early_stopping, lr_reducer], validation_split =0.2 )


        scores = [history.history['val_loss'][epoch] for epoch in range(len(history.history['loss']))]



        score = min(scores)
        #global best_score, best_model
        if self.best_score > score:
                self.best_score = score
                self.best_model = model

        end_time = perf_counter()


        return {'loss': score, 'status': STATUS_OK, 'time': end_time - start_time}




    def evaluate_model_timeNet(self,model,embedding_size):
        #models = load_model("models/generate_" + self.log_name + ".h5")

        X_test, t_test = self.build_windows_timeNet(self.traces_test,self.seconds_traces_test,self.win_size)

        # if(self.net_out!=2):
        #     print(np.unique(Y_test))
        #     #Y_test = self.leA.transform(Y_test.reshape(-1,1))
        #     Y_test= self.leA.transform(Y_test)
        #     print(np.unique(Y_test))
        #     #print(Y_test)
        #     #print(np.unique(Y_test))
        # if(self.net_out!=1):
        #     Z_test = self.leO.transform(Z_test)
        # #print(Y_test)

        if(self.net_embedding==1):
             #w2v_modelX.save("models/w2v/generate_"+ self.log_name+ "_size" + str(size) + ".h5")

            w2v_model = Word2Vec.load("models/w2v/generate_"+ self.log_name+ "_size" + str(embedding_size) + ".h5")
            self.save_w2v_vocab(w2v_model)
            X_test = self.encodePrefixes(embedding_size,X_test)
            # X_test = np.asarray()


        prediction = model.predict(X_test, batch_size=128, verbose = 0)

        print(prediction)
        mae = metrics.mean_absolute_error(t_test, prediction)
        return mae
        # if(self.net_out==0):
        #     y_pred = prediction[0]
        #     z_pred = prediction[1]
        # elif(self.net_out==1):
        #     y_pred = prediction
        # elif(self.net_out==2):
        #     z_pred = prediction
        #
        #
        # if(self.net_out!=2):
        #     rounded_act_prediction = np.argmax(y_pred,axis=-1)
        #
        #     # print(rounded_act_prediction)
        #     # print(np.unique(rounded_act_prediction))
        #     #print(rounded_act_prediction)
        #     cm_act = confusion_matrix(y_true= Y_test, y_pred=rounded_act_prediction)
        #     cm_plot_labels_act = range(0,self.outsize_act)
        #     #print(cm_act)
        #     self.plot_confusion_matrix(cm=cm_act, classes=cm_plot_labels_act, title='Confusion Matrix Next Activity')
        #     reportNA=metrics.classification_report(Y_test, rounded_act_prediction, digits=3)
        #     print(reportNA)
        #     if(self.net_out==1):
        #         return reportNA,cm_act
        #
        # if(self.net_out!=1):
        #     rounded_out_prediction = np.argmax(z_pred,axis=-1)
        #     cm_out = confusion_matrix(y_true= Z_test, y_pred=rounded_out_prediction)
        #     cm_plot_labels_out = range(0,self.outsize_out)
        #     #print(cm_out)
        #     self.plot_confusion_matrix(cm=cm_out, classes=cm_plot_labels_out, title='Confusion Matrix Outcome')
        #     reportO = metrics.classification_report(Z_test, rounded_out_prediction, digits=3)
        #     print(reportO)
        #     if(self.net_out==2):
        #         return reportO,cm_out
        #
        # return reportNA,cm_act,reportO,cm_out




    # def evaluate_model_nextAct(self,models, win_size):
    #     X_test, Y_test, _ = self.build_windows(self.traces_test,win_size)
    #     Y_test = self.leA.transform(Y_test)
    #
    #     prediction = models.predict(X_test, batch_size=128, verbose = 0)
    #     y_pred=prediction
    #     rounded_act_prediction = np.argmax(y_pred,axis=-1)
    #
    #     cm_act = confusion_matrix(y_true= Y_test, y_pred=rounded_act_prediction)
    #     cm_plot_labels_act = range(0,self.outsize_act)
    #     #print(cm_act)
    #     self.plot_confusion_matrix(cm=cm_act, classes=cm_plot_labels_act, title='Confusion Matrix Next Activity')
    #     reportNA=metrics.classification_report(Y_test, rounded_act_prediction, digits=3)
    #     print(reportNA)
    #     return reportNA,cm_act

    # def evaluate_model_outcome(self,models, win_size):
    #     X_test, _, Z_test = self.build_windows(self.traces_test,win_size)
    #     Z_test = self.leO.transform(Z_test)
    #
    #     prediction = models.predict(X_test, batch_size=128, verbose = 0)
    #     z_pred = prediction
    #
    #
    #     rounded_out_prediction = np.argmax(z_pred,axis=-1)
    #
    #     cm_out = confusion_matrix(y_true= Z_test, y_pred=rounded_out_prediction)
    #     cm_plot_labels_out = range(0,self.outsize_out)
    #     #print(cm_out)
    #     self.plot_confusion_matrix(cm=cm_out, classes=cm_plot_labels_out, title='Confusion Matrix Outcome')
    #     reportO = metrics.classification_report(Z_test, rounded_out_prediction, digits=3)
    #     print(reportO)
    #
    #     return reportO,cm_out



    pass

#
# def build_w2v(self):
# '''
# :return: mapping parola -> vettore per il modello reset
# '''
# temp_traces = []
# for k in self.prefix_list:
#     listToStr = ' '.join([self.replace_char(str(elem)) for elem in k[1]])
#     temp_traces.append(listToStr) self.temp_label_int = []
# for l in self.next_act_list:
#     self.temp_label_int.append(l[1]) tokenized_words = []
# for s in temp_traces:
#     tokenized_words.append(s.split(' ')) # inizializzo W2V con il parametro ottimizzato self.vec_dim
#     self.w2v_model = Word2Vec(vector_size=self.vec_dim, seed=seed, min_count=1, sg=0, workers=1)
#     self.w2v_model.build_vocab(tokenized_words, min_count=1)
#     total_examples = self.w2v_model.corpus_count
#
# # addestro W2V
# self.w2v_model.train(tokenized_words, total_examples=total_examples, epochs=200)
# # salvo modello W2V
# self.w2v_model.save("w2v/" + self.log_name + "/generate_" + self.log_name + '_' + str(self.count_model) + ".h5")
# print('new W2V models-->',
# "w2v/" + self.log_name + "/generate_" + self.log_name + '_' + str(self.count_model) + ".h5")
# vocab = list(self.w2v_model.wv.index_to_key) print("The total number of words are : ", len(vocab))
# print("The no of words :", len(vocab))
# print("The words in vocab :", (vocab)) self.word_vec_dict = {}
# for word in vocab:
# self.word_vec_dict[word] = self.w2v_model.wv.get_vector(word) # estraggo trace di lunghezza pari a 4
# temp_traces_int = []
# for l in temp_traces:
# l = l.split()
# win = l[-4:]
# temp_traces_int.append(win)
# temp_traces_int = np.asarray(temp_traces_int)
# ###################################### pad_rev = pad_sequences(temp_traces_int, maxlen=4, padding='pre', dtype=object, value='_PAD_') self.list_emb = []
# for l in pad_rev:
# list_emb_temp = []
# for t in l:
# embed_vector = self.word_vec_dict.get(t)
# if embed_vector is not None: # word is in the vocabulary learned by the w2v models
# list_emb_temp.append(embed_vector)
# else:
# list_emb_temp.append(np.zeros(shape=(self.vec_dim)))
# self.list_emb.append(list_emb_temp)
# self.list_emb = np.asarray(self.list_emb) # salvo il nuovo modello W2V
# output = open("w2v_dict/" + self.log_name + "/generate_" + self.log_name + '_' + str(self.count_model) + ".pkl",
# 'wb')
# pickle.dump(self.word_vec_dict, output)
# output.close()
#
#
#
