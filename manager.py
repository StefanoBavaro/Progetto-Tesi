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

    def __init__(self, log_name, act_name, case_name, ts_name, out_name, ex_size):
        self.log_name = log_name
        self.timestamp_name = ts_name
        self.case_name = case_name
        self.activity_name = act_name
        self.outcome_name = out_name

        self.act_dictionary = {}
        self.traces = [[]]
        self.example_size = ex_size

        self.x_training=[]
        self.y_training=[]
        self.z_training=[]

        self.outsize_act=0
        self.outsize_out=0


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
        return self.build_training_test_sets()

    def getTraces(self,data):
        """
        Codifica ogni traccia in sequenze di interi (ovvero le codifiche di ogni attività)
        """
        self.traces = np.empty((len(data[self.case_name].unique()),), dtype=object)
        self.traces[...]=[[] for _ in range(len(data[self.case_name].unique()))]
        outcomes = range(1, len(data[self.outcome_name].unique())+1)
        #print(outcomes)

        traces_counter = 0
        for i, event in data.iterrows():
            activity_coded = self.act_dictionary[event[self.activity_name]]
            #print(activity_coded)
            self.traces[traces_counter].append(activity_coded)
            #print(self.traces[traces_counter])
            if(activity_coded in outcomes):
                traces_counter+=1

        #converto ogni traccia in array piuttosto che liste
        for i in range(0,len(self.traces)):
            self.traces[i] = np.array(self.traces[i])

        #print(self.traces.size)
        #print(self.traces)


    def build_training_test_sets(self):
        """
        Genera finestre di lunghezza fissata (self.example_size) per ogni traccia. (self.x_training)
        Genera un array contenente le next activities (interi) per ogni finestra. (self.y_training)
        Genera un array contenente l'outcome (intero) per ogni finestra. (self.z_training)
        """
        #print(self.traces)

        #print(self.traces.size)
        for trace in self.traces:
            #print(trace)
            i=0
            #print(i)
            j=1
            while j < trace.size:
                #print(j)
                current_example = np.zeros(self.example_size)
                values = trace[0:j] if j <= self.example_size else \
                         trace[j - self.example_size:j]
                self.y_training.append(trace[j])
                current_example[self.example_size - values.size:] = values
                self.x_training.append(current_example)
                encoded_outcome = trace[trace.size-1]
                self.z_training.append(encoded_outcome)
                # if(j==(trace.size-1)):
                #     self.z_training.append(trace[j])
                j += 1
            i+=1

        # print(self.x_training)
        # print(len(self.x_training))
        # print(self.y_training)
        # print(len(self.y_training))
        # print(self.z_training)
        # print(len(self.z_training))

        self.y_training = np.asarray(self.y_training)
        self.z_training = np.asarray(self.z_training)

        le = preprocessing.LabelEncoder()
        self.y_training = le.fit_transform(self.y_training)
        self.z_training = le.fit_transform(self.z_training)

        # print("After first transformation:")
        # print(self.x_training)
        # print(self.y_training)
        # print(self.z_training)

        self.x_training = np.asarray(self.x_training)

        # print("After second transformation:")
        # print(self.x_training)
        # print(self.y_training)
        # print(self.z_training)
        # print(self.act_dictionary)

        X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(self.x_training, self.y_training, self.z_training, test_size=0.2,
                                                          random_state=42, shuffle=True)

        return X_train, X_test, Y_train, Y_test, Z_train, Z_test



    def build_neural_network_model(self,X_train, Y_train, Z_train ):


        Y_train = to_categorical(Y_train)
        Z_train = to_categorical(Z_train) #trasformazione in vettori per la rete

        unique_events = len(self.act_dictionary) #numero di diversi eventi/attività nel dataset
        # X_train, X_val, Y_train, Y_val, Z_train, Z_val = train_test_split(self.x_training, self.y_training, self.z_training, test_size=0.2,
        #                                                   random_state=42, shuffle=True)

        # print(Y_train.shape)
        # print(Z_train.shape)
        size_act = (unique_events + 1) // 2

        input_act = Input(shape=(self.example_size,), dtype='int32', name='input_act')
        x_act = Embedding(output_dim=size_act, input_dim=unique_events+1, input_length=self.example_size)(
                         input_act)

        l1 = LSTM(16, return_sequences=True, kernel_initializer='glorot_uniform')(x_act)
        b1 = BatchNormalization()(l1)
        l2_1 = LSTM(16, return_sequences=False, kernel_initializer='glorot_uniform')(b1) # the layer specialized in activity prediction
        b2_1 = BatchNormalization()(l2_1)
        l2_2 = LSTM(16, return_sequences=False, kernel_initializer='glorot_uniform')(b1) #the layer specialized in outcome prediction
        b2_2 = BatchNormalization()(l2_2)

        output_l = Dense(self.outsize_act, activation='softmax', name='act_output')(b2_1)
        output_o = Dense(self.outsize_out, activation='softmax', name='outcome_output')(b2_2)

        model = Model(inputs=input_act, outputs=[output_l, output_o])
        print(model.summary())

        opt = Adam()
        model.compile(loss={'act_output':'categorical_crossentropy', 'outcome_output':'categorical_crossentropy'}, optimizer=opt, metrics=['accuracy'])
        # early_stopping = EarlyStopping(monitor='val_loss',
        #                                patience=20)
        # lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto',
        #                                min_delta=0.0001, cooldown=0, min_lr=0)

        early_stopping = EarlyStopping(monitor='val_loss', patience=42)
        #model_checkpoint = ModelCheckpoint('output_files/models/model_{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
        lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

        #Y_train = tf.squeeze(Y_train, axis=-1)
        #Z_train = tf.squeeze(Z_train, axis=-1)
        model.fit(X_train, [Y_train, Z_train], epochs=500, batch_size=128, verbose=2, callbacks=[early_stopping, lr_reducer], validation_split =0.2 )
        #validation_data=(X_val, [Y_val,Z_val])

        model.save("model/generate_" + self.log_name + ".h5")



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
        plt.show()

    def evaluate_model(self,X_test,Y_test,Z_test):
        model = load_model("model/generate_" + self.log_name + ".h5")

        prediction = model.predict(X_test, batch_size=128, verbose = 0)
        y_pred = prediction[0]
        z_pred = prediction[1]

        # accuracy = accuracy_score([Y_test,Z_test], prediction)
        # print(accuracy)
        # Y_accuracy = accuracy_score(Y_test, y_pred)
        # Z_accuracy = accuracy_score(Z_test, z_pred)
        #return {'loss': -accuracy, 'status': STATUS_OK}

        # search_space = hp.randint('n_estimators', 200, 1000)
        #
        # space = {'lstmsize': scope.int(hp.loguniform('lstmsize', np.log(10), np.log(150))),
        #  'dropout': hp.uniform("dropout", 0, 0.5),
        #  'l1': hp.loguniform("l1", np.log(0.00001), np.log(0.1)),
        #  'l2': hp.loguniform("l2", np.log(0.00001), np.log(0.1)),
        #  'batch_size': scope.int(hp.uniform('batch_size', 3, 6)),
        #  'learning_rate': hp.loguniform("learning_rate", np.log(0.00001), np.log(0.01)),
        #  'n_layers': hp.choice('n_layers', [
        #      {'n_layers': 1},
        #      {'n_layers': 2,
        #       'lstmsize22': scope.int(hp.loguniform('lstmsize22', np.log(10), np.log(150)))},
        #      {'n_layers': 3,
        #       'lstmsize32': scope.int(hp.loguniform('lstmsize32', np.log(10), np.log(150))),
        #       'lstmsize33': scope.int(hp.loguniform('lstmsize33', np.log(10), np.log(150)))}
        #  ])
        # }





        rounded_act_prediction = np.argmax(y_pred,axis=-1)
        rounded_out_prediction = np.argmax(z_pred,axis=-1)
        # print(prediction)
        # print(rounded_act_prediction)
        # print(rounded_out_prediction)

        cm_act = confusion_matrix(y_true= Y_test, y_pred=rounded_act_prediction)
        cm_plot_labels_act = range(0,self.outsize_act)
        #print(cm_act)
        self.plot_confusion_matrix(cm=cm_act, classes=cm_plot_labels_act, title='Confusion Matrix Next Activity')
        print(metrics.classification_report(Y_test, rounded_act_prediction, digits=3))

        cm_out = confusion_matrix(y_true= Z_test, y_pred=rounded_out_prediction)
        cm_plot_labels_out = range(0,self.outsize_out)
        #print(cm_out)
        self.plot_confusion_matrix(cm=cm_out, classes=cm_plot_labels_out, title='Confusion Matrix Outcome')
        print(metrics.classification_report(Z_test, rounded_out_prediction, digits=3))





    pass
