from manager import Manager
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK

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
from hyperopt.pyll.base import scope
from hyperopt.pyll.stochastic import sample



log_name="Production_Sorted"
activity_name = "Activity"
case_name = "Case ID"
timestamp_name = "Complete Timestamp"
outcome_name = 'label'
example_size = 4

manager = Manager(log_name, activity_name, case_name, timestamp_name, outcome_name, example_size)
manager.gen_internal_csv()

X_train, X_test, Y_trainBefore, Y_test, Z_trainBefore, Z_test = manager.csv_to_data()
# manager.build_neural_network_model(X_train,Y_train,Z_train)
# manager.evaluate_model(X_test,Y_test,Z_test)


def objective(params):
        Y_train = to_categorical(Y_trainBefore)
        Z_train = to_categorical(Z_trainBefore)

        unique_events = len(manager.act_dictionary) #numero di diversi eventi/attività nel dataset
        size_act = (unique_events + 1) // 2

        input_act = Input(shape=(manager.example_size,), dtype='int32', name='input_act')
        x_act = Embedding(output_dim= params["output_dim_embedding"], input_dim=unique_events + 1, input_length=manager.example_size)(
                         input_act)

        l1 = LSTM(params["lstm1_size"], return_sequences=True, kernel_initializer='glorot_uniform')(x_act)
        b1 = BatchNormalization()(l1)
        l2_1 = LSTM(params["lstm2_size"], return_sequences=False, kernel_initializer='glorot_uniform')(b1) # the layer specialized in activity prediction
        b2_1 = BatchNormalization()(l2_1)
        l2_2 = LSTM(params["lstm3_size"], return_sequences=False, kernel_initializer='glorot_uniform')(b1) #the layer specialized in outcome prediction
        b2_2 = BatchNormalization()(l2_2)

        output_l = Dense(manager.outsize_act, activation='softmax', name='act_output')(b2_1)
        output_o = Dense(manager.outsize_out, activation='softmax', name='outcome_output')(b2_2)

        model = Model(inputs=input_act, outputs=[output_l, output_o])
        print(model.summary())

        opt = Adam(lr=params["learning_rate"])
        model.compile(loss={'act_output':'categorical_crossentropy', 'outcome_output':'categorical_crossentropy'}, optimizer=opt, metrics=['accuracy'])
        # early_stopping = EarlyStopping(monitor='val_loss',
        #                                patience=20)
        # lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto',
        #                                min_delta=0.0001, cooldown=0, min_lr=0)

        early_stopping = EarlyStopping(monitor='val_loss', patience=42)
        #model_checkpoint = ModelCheckpoint('output_files/models/model_{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
        lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

        history = model.fit(X_train, [Y_train, Z_train], epochs=500, batch_size=2**params['batch_size'], verbose=2, callbacks=[early_stopping, lr_reducer], validation_split =0.2 )
        scores = [history.history['val_loss'][epoch] for epoch in range(len(history.history['loss']))]
        score = min(scores)
        global best_score, best_model
        if best_score > score:
                best_score = score
                best_model = model
        # prediction = model.predict(X_test, batch_size=128, verbose = 0)
        # y_pred = prediction[0]
        # z_pred = prediction[1]
        # accuracy_act = accuracy_score(Y_test, np.argmax(y_pred,axis=-1))
        # accuracy_out = accuracy_score(Z_test, np.argmax(z_pred,axis=-1))

        #
        #
        # accuracy_score(testing['crime'], predicted2.argmax(axis=1))
        #
        # rounded_act_prediction = np.argmax(y_pred,axis=-1)
        # rounded_out_prediction = np.argmax(z_pred,axis=-1)
        #return {'loss_act': -accuracy_act, 'loss_out': -accuracy_out,'status': STATUS_OK}
        return {'loss': score, 'status': STATUS_OK}
        #model.save("model/generate_" + self.log_name + ".h5")

search_space = {'output_dim_embedding':hp.randint('output_dim_embedding',4,1000),
                'lstm1_size': scope.int(hp.loguniform('lstm1_size', np.log(10), np.log(150))),
                'lstm2_size': scope.int(hp.loguniform('lstm2_size', np.log(10), np.log(150))),
                'lstm3_size': scope.int(hp.loguniform('lstm3_size', np.log(10), np.log(150))),
                'batch_size': scope.int(hp.uniform('batch_size', 3, 6)),
                'learning_rate': hp.loguniform("learning_rate", np.log(0.00001), np.log(0.01)),
                }
algorithm = tpe.suggest

best_score = np.inf
best_model = None
trials = Trials()


best_params = fmin(
  fn=objective,
  space=search_space,
  algo=algorithm,
  max_evals=200,
  trials=trials)

print(best_params)

#manager.objective(X_train, X_test, Y_train, Y_test, Z_train, Z_test)





