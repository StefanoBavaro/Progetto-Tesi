

import numpy as np
import pandas as pd
import datetime
from pathlib import Path
import itertools

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
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK


class Manager:

    def __init__(self, log_name, act_name, case_name, ts_name, out_name):
        self.log_name = log_name
        self.timestamp_name = ts_name
        self.case_name = case_name
        self.activity_name = act_name
        self.outcome_name = out_name

        self.act_dictionary = {}
        self.traces = [[]]
        self.traces_train = []
        self.traces_test = []
        #self.example_size = ex_size

        self.x_training=[]
        self.y_training=[]
        self.z_training=[]

        self.best_score = np.inf
        self.best_model = None

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
        data = pd.read_csv(filename)


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
        print("NUMERO EVENTI DOPO UPDATE= " + str(num_events))

        self.outsize_act = len(data_updated[self.activity_name].unique())
        print(self.outsize_act)
        self.outsize_out = len(data_updated[self.outcome_name].unique())
        print(self.outsize_out)

        filename = str(Path('../Progetto-Tesi/data/updated_csvs/'+self.log_name+"_updated.csv").resolve())
        data_updated.to_csv(filename)
        #print(data_updated)
        print("Done: updated CSV created")

    def csv_to_data(self):
        """
        Legge il file csv con le tracce aggiornate e costruisce un dizionario attività-chiave, dove la chiave è un intero
        unico con cui verranno codificate le tracce.
        Richiama la funzione di codifica delle tracce e della costruzione delle finestre.
        """

        filename = str(Path('../Progetto-Tesi/data/updated_csvs/' + self.log_name + "_updated.csv").resolve())
        data = pd.read_csv(filename)

        #inserisco per prime nel dizionario le label degli outcome
        outcomes = data[self.outcome_name].unique()
        for i in range(0,len(outcomes)):
            self.act_dictionary[outcomes[i]] = i+1

        #inserisco il resto delle attività nel dizionario
        for i, event in data.iterrows():
            if event[self.activity_name] not in self.act_dictionary.keys():
                self.act_dictionary[event[self.activity_name]] = len(self.act_dictionary.keys()) + 1

        #print(self.act_dictionary)

        self.getTraces(data)
        #self.build_training_test_sets()
        #return self.build_training_test_sets()

    def getTraces(self,data):
        """
        Codifica ogni traccia in sequenze di interi (ovvero le codifiche di ogni attività)
        """
        traces = np.empty((len(data[self.case_name].unique()),), dtype=object)
        traces[...]=[[] for _ in range(len(data[self.case_name].unique()))]
        outcomes = range(1, len(data[self.outcome_name].unique())+1)
        #print(outcomes)

        traces_counter = 0
        for i, event in data.iterrows():
            activity_coded = self.act_dictionary[event[self.activity_name]]
            #print(activity_coded)
            traces[traces_counter].append(activity_coded)
            #print(self.traces[traces_counter])
            if(activity_coded in outcomes):
                traces_counter+=1

        #converto ogni traccia in array piuttosto che liste
        for i in range(0,len(traces)):
            traces[i] = np.array(traces[i])


        self.traces_train, self.traces_test = train_test_split(traces, test_size=0.2, random_state=42, shuffle=False)


    def build_windows(self,traces,win_size):
        X_vec = []
        Y_vec = []
        Z_vec = []

        for trace in traces:
            #print(trace)
            i=0
            #print(i)
            j=1
            while j < trace.size:
                #print(j)
                current_example = np.zeros(win_size)
                values = trace[0:j] if j <= win_size else \
                         trace[j - win_size:j]
                Y_vec.append(trace[j])
                current_example[win_size - values.size:] = values
                X_vec.append(current_example)
                encoded_outcome = trace[trace.size-1]
                Z_vec.append(encoded_outcome)
                j += 1
            i+=1

        X_vec = np.asarray(X_vec)
        Y_vec = np.asarray(Y_vec)
        Z_vec = np.asarray(Z_vec)

        return X_vec,Y_vec,Z_vec

    # def build_training_test_sets(self):
    #     """
    #     Genera finestre di lunghezza fissata (self.example_size) per ogni traccia. (self.x_training)
    #     Genera un array contenente le next activities (interi) per ogni finestra. (self.y_training)
    #     Genera un array contenente l'outcome (intero) per ogni finestra. (self.z_training)
    #     """
    #     #print(self.traces)
    #
    #     #traces_train, traces_test = train_test_split(self.traces, test_size=0.2, random_state=42, shuffle=False)
    #
    #     # print(len(self.traces))
    #     # print(len(traces_train))
    #     # print(len(traces_test))
    #
    #     X_train = []
    #     Y_train = []
    #     Z_train = []
    #
    #     X_test = []
    #     Y_test = []
    #     Z_test = []
    #
    #     #generate training sets
    #     for trace in traces_train:
    #         #print(trace)
    #         i=0
    #         #print(i)
    #         j=1
    #         while j < trace.size:
    #             #print(j)
    #             current_example = np.zeros(self.example_size)
    #             values = trace[0:j] if j <= self.example_size else \
    #                      trace[j - self.example_size:j]
    #             Y_train.append(trace[j])
    #             current_example[self.example_size - values.size:] = values
    #             X_train.append(current_example)
    #             encoded_outcome = trace[trace.size-1]
    #             Z_train.append(encoded_outcome)
    #             j += 1
    #         i+=1
    #
    #     for trace in traces_test:
    #         #print(trace)
    #         i=0
    #         #print(i)
    #         j=1
    #         while j < trace.size:
    #             #print(j)
    #             current_example = np.zeros(self.example_size)
    #             values = trace[0:j] if j <= self.example_size else \
    #                      trace[j - self.example_size:j]
    #             Y_test.append(trace[j])
    #             current_example[self.example_size - values.size:] = values
    #             X_test.append(current_example)
    #             encoded_outcome = trace[trace.size-1]
    #             Z_test.append(encoded_outcome)
    #             j += 1
    #         i+=1
    #
    #     # print(self.x_training)
    #     # print(len(self.x_training))
    #     # print(self.y_training)
    #     # print(len(self.y_training))
    #     # print(self.z_training)
    #     # print(len(self.z_training))
    #
    #
    #     X_train = np.asarray(X_train)
    #     Y_train = np.asarray(Y_train)
    #     Z_train = np.asarray(Z_train)
    #
    #     X_test = np.asarray(X_test)
    #     Y_test = np.asarray(Y_test)
    #     Z_test = np.asarray(Z_test)
    #
    #     # print("After first transformation:")
    #     # print(self.x_training)
    #     # print(self.y_training)
    #     # print(self.z_training)
    #
    #     # print("After second transformation:")
    #     # print(self.x_training)
    #     # print(self.y_training)
    #     # print(self.z_training)
    #     # print(self.act_dictionary)
    #
    #     # X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(self.x_training, self.y_training, self.z_training, test_size=0.2,
    #     #                                                   random_state=42, shuffle=False)
    #
    #     leA = preprocessing.LabelEncoder()
    #     Y_train = leA.fit_transform(Y_train)
    #     Y_test = leA.transform(Y_test)
    #
    #     leO = preprocessing.LabelEncoder()
    #     Z_train = leO.fit_transform(Z_train)
    #     Z_test = leO.transform(Z_test)
    #
    #     return X_train, X_test, Y_train, Y_test, Z_train, Z_test

    def objective(self,params):

            unique_events = len(self.act_dictionary) #numero di diversi eventi/attività nel dataset
            #size_act = (unique_events + 1) // 2

            input_act = Input(shape=(params['win_size'],), dtype='int32', name='input_act')
            x_act = Embedding(output_dim=params["output_dim_embedding"], input_dim=unique_events + 1, input_length=params['win_size'])(
                             input_act)
            #gensim per word to vec
            n_layers = int(params["n_layers"]["n_layers"])

            l1 = LSTM(params["shared_lstm_size"], return_sequences=True, kernel_initializer='glorot_uniform',dropout=params['dropout'])(x_act)
            l1 = BatchNormalization()(l1)

            l_a = LSTM(params["lstmA_size_1"], return_sequences=(n_layers != 1), kernel_initializer='glorot_uniform',dropout=params['dropout'])(l1)
            l_a = BatchNormalization()(l_a)
            l_o = LSTM(params["lstmO_size_1"], return_sequences=(n_layers != 1), kernel_initializer='glorot_uniform',dropout=params['dropout'])(l1)
            l_o = BatchNormalization()(l_o)

            for i in range(2,n_layers+1):
                l_a = LSTM(params["n_layers"]["lstmA_size_%s_%s" % (i, n_layers)], return_sequences=(n_layers != i), kernel_initializer='glorot_uniform',dropout=params['dropout'])(l_a)
                l_a = BatchNormalization()(l_a)

                l_o = LSTM(params["n_layers"]["lstmO_size_%s_%s" % (i, n_layers)], return_sequences=(n_layers != i), kernel_initializer='glorot_uniform',dropout=params['dropout'])(l_o)
                l_o = BatchNormalization()(l_o)

            output_l = Dense(self.outsize_act, activation='softmax', name='act_output')(l_a)
            output_o = Dense(self.outsize_out, activation='softmax', name='outcome_output')(l_o)

            model = Model(inputs=input_act, outputs=[output_l, output_o])
            print(model.summary())

            opt = Adam(lr=params["learning_rate"])

            model.compile(loss={'act_output':'categorical_crossentropy', 'outcome_output':'categorical_crossentropy'}, optimizer=opt, loss_weights=[params['gamma'], 1-params['gamma']] ,metrics=['accuracy'])
            early_stopping = EarlyStopping(monitor='val_loss',
                                           patience=20)
            lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto',
                                           min_delta=0.0001, cooldown=0, min_lr=0)


            X_train,Y_train,Z_train = self.build_windows(self.traces_train,params['win_size'])

            Y_train = self.leA.fit_transform(Y_train)
            Z_train = self.leO.fit_transform(Z_train)

            Y_train = to_categorical(Y_train)
            Z_train = to_categorical(Z_train)

            history = model.fit(X_train, [Y_train, Z_train], epochs=500, batch_size=2**params['batch_size'], verbose=2, callbacks=[early_stopping, lr_reducer], validation_split =0.2 )

            scores = [history.history['val_loss'][epoch] for epoch in range(len(history.history['loss']))]

            score = min(scores)
            #global best_score, best_model
            if self.best_score > score:
                    self.best_score = score
                    self.best_model = model

            return {'loss': score, 'status': STATUS_OK}
            #model.save("model/generate_" + self.log_name + ".h5")

    def nextActivityNetwork(self,params):

            unique_events = len(self.act_dictionary) #numero di diversi eventi/attività nel dataset
            #size_act = (unique_events + 1) // 2

            input_act = Input(shape=(params['win_size'],), dtype='int32', name='input_act')
            x_act = Embedding(output_dim=params["output_dim_embedding"], input_dim=unique_events + 1, input_length=params['win_size'])(
                             input_act)
                #gensim per word to vec
            n_layers = int(params["n_layers"]["n_layers"])

            l_a = LSTM(params["lstmA_size_1"], return_sequences=(n_layers != 1), kernel_initializer='glorot_uniform',dropout=params['dropout'])(x_act)
            l_a = BatchNormalization()(l_a)

            for i in range(2,n_layers+1):
                l_a = LSTM(params["n_layers"]["lstmA_size_%s_%s" % (i, n_layers)], return_sequences=(n_layers != i), kernel_initializer='glorot_uniform',dropout=params['dropout'])(l_a)
                l_a = BatchNormalization()(l_a)

            output_l = Dense(self.outsize_act, activation='softmax', name='act_output')(l_a)
            model = Model(inputs=input_act, outputs=output_l)

            print(model.summary())

            opt = Adam(lr=params["learning_rate"])

            model.compile(loss={'act_output':'categorical_crossentropy'}, optimizer=opt,metrics=['accuracy'])
            early_stopping = EarlyStopping(monitor='val_loss',patience=20)
            lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto',
                                               min_delta=0.0001, cooldown=0, min_lr=0)


            X_train,Y_train,_ = self.build_windows(self.traces_train,params['win_size'])

            Y_train = self.leA.fit_transform(Y_train)
            Y_train = to_categorical(Y_train)

            history = model.fit(X_train, Y_train, epochs=500, batch_size=2**params['batch_size'], verbose=2, callbacks=[early_stopping, lr_reducer], validation_split =0.2 )

            scores = [history.history['val_loss'][epoch] for epoch in range(len(history.history['loss']))]

            score = min(scores)
                #global best_score, best_model
            if self.best_score > score:
                    self.best_score = score
                    self.best_model = model

            return {'loss': score, 'status': STATUS_OK}
            #model.save("model/generate_" + self.log_name + ".h5")

    def objective(self,params):

            unique_events = len(self.act_dictionary) #numero di diversi eventi/attività nel dataset
            #size_act = (unique_events + 1) // 2

            input_act = Input(shape=(params['win_size'],), dtype='int32', name='input_act')
            x_act = Embedding(output_dim=params["output_dim_embedding"], input_dim=unique_events + 1, input_length=params['win_size'])(
                             input_act)
            #gensim per word to vec
            n_layers = int(params["n_layers"]["n_layers"])

            l_o = LSTM(params["lstmO_size_1"], return_sequences=(n_layers != 1), kernel_initializer='glorot_uniform',dropout=params['dropout'])(x_act)
            l_o = BatchNormalization()(l_o)

            for i in range(2,n_layers+1):
                l_o = LSTM(params["n_layers"]["lstmO_size_%s_%s" % (i, n_layers)], return_sequences=(n_layers != i), kernel_initializer='glorot_uniform',dropout=params['dropout'])(l_o)
                l_o = BatchNormalization()(l_o)

            output_o = Dense(self.outsize_out, activation='softmax', name='outcome_output')(l_o)

            model = Model(inputs=input_act, outputs=output_o)
            print(model.summary())

            opt = Adam(lr=params["learning_rate"])

            model.compile(loss={'outcome_output':'categorical_crossentropy'}, optimizer=opt, metrics=['accuracy'])
            early_stopping = EarlyStopping(monitor='val_loss',patience=20)
            lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto',
                                           min_delta=0.0001, cooldown=0, min_lr=0)


            X_train,_,Z_train = self.build_windows(self.traces_train,params['win_size'])

            Z_train = self.leO.fit_transform(Z_train)
            Z_train = to_categorical(Z_train)

            history = model.fit(X_train, Z_train, epochs=500, batch_size=2**params['batch_size'], verbose=2, callbacks=[early_stopping, lr_reducer], validation_split =0.2 )

            scores = [history.history['val_loss'][epoch] for epoch in range(len(history.history['loss']))]

            score = min(scores)
            #global best_score, best_model
            if self.best_score > score:
                    self.best_score = score
                    self.best_model = model

            return {'loss': score, 'status': STATUS_OK}
            #model.save("model/generate_" + self.log_name + ".h5")

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


    def evaluate_model_nextAct(self,model, win_size):
        X_test, Y_test, _ = self.build_windows(self.traces_test,win_size)
        Y_test = self.leA.transform(Y_test)

        prediction = model.predict(X_test, batch_size=128, verbose = 0)
        y_pred=prediction[0]
        rounded_act_prediction = np.argmax(y_pred,axis=-1)

        cm_act = confusion_matrix(y_true= Y_test, y_pred=rounded_act_prediction)
        cm_plot_labels_act = range(0,self.outsize_act)
        #print(cm_act)
        self.plot_confusion_matrix(cm=cm_act, classes=cm_plot_labels_act, title='Confusion Matrix Next Activity')
        reportNA=metrics.classification_report(Y_test, rounded_act_prediction, digits=3)
        print(reportNA)
        return reportNA,cm_act

    def evaluate_model_outcome(self,model, win_size):
        X_test, _, Z_test = self.build_windows(self.traces_test,win_size)
        Z_test = self.leO.transform(Z_test)

        prediction = model.predict(X_test, batch_size=128, verbose = 0)
        if(len(prediction) == 1):
            z_pred = prediction[0]
        else:
            z_pred = prediction[1]

        rounded_out_prediction = np.argmax(z_pred,axis=-1)

        cm_out = confusion_matrix(y_true= Z_test, y_pred=rounded_out_prediction)
        cm_plot_labels_out = range(0,self.outsize_out)
        #print(cm_out)
        self.plot_confusion_matrix(cm=cm_out, classes=cm_plot_labels_out, title='Confusion Matrix Outcome')
        reportO = metrics.classification_report(Z_test, rounded_out_prediction, digits=3)
        print(reportO)

        return reportO,cm_out

    def evaluate_model(self,win_size):
        model = load_model("model/generate_" + self.log_name + ".h5")

        X_test, Y_test, Z_test = self.build_windows(self.traces_test,win_size)
        Y_test = self.leA.transform(Y_test)
        Z_test = self.leO.transform(Z_test)

        prediction = model.predict(X_test, batch_size=128, verbose = 0)
        y_pred = prediction[0]
        z_pred = prediction[1]

        rounded_act_prediction = np.argmax(y_pred,axis=-1)
        rounded_out_prediction = np.argmax(z_pred,axis=-1)
        # print(prediction)
        # print(rounded_act_prediction)
        # print(rounded_out_prediction)

        cm_act = confusion_matrix(y_true= Y_test, y_pred=rounded_act_prediction)
        cm_plot_labels_act = range(0,self.outsize_act)
        #print(cm_act)
        self.plot_confusion_matrix(cm=cm_act, classes=cm_plot_labels_act, title='Confusion Matrix Next Activity')
        reportNA=metrics.classification_report(Y_test, rounded_act_prediction, digits=3)
        print(reportNA)

        cm_out = confusion_matrix(y_true= Z_test, y_pred=rounded_out_prediction)
        cm_plot_labels_out = range(0,self.outsize_out)
        #print(cm_out)
        self.plot_confusion_matrix(cm=cm_out, classes=cm_plot_labels_out, title='Confusion Matrix Outcome')
        reportO = metrics.classification_report(Z_test, rounded_out_prediction, digits=3)
        print(reportO)

        return reportNA,cm_act,reportO,cm_out





    pass

