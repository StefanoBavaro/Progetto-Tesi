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
from keras.layers import Embedding, Dense, BatchNormalization, Reshape, CuDNNLSTM
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
outcome_name = "label"
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
        x_act = Embedding(output_dim=params["output_dim_embedding"], input_dim=unique_events + 1, input_length=manager.example_size)(
                         input_act)

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


        # l1 = LSTM(params["lstm1_size"], return_sequences=True, kernel_initializer='glorot_uniform')(x_act)
        # b1 = BatchNormalization()(l1)
        # l2_1 = LSTM(params["lstm2_size"], return_sequences=False, kernel_initializer='glorot_uniform')(b1) # the layer specialized in activity prediction
        # b2_1 = BatchNormalization()(l2_1)
        # l2_2 = LSTM(params["lstm3_size"], return_sequences=False, kernel_initializer='glorot_uniform')(b1) #the layer specialized in outcome prediction
        # b2_2 = BatchNormalization()(l2_2)

        output_l = Dense(manager.outsize_act, activation='softmax', name='act_output')(l_a)
        output_o = Dense(manager.outsize_out, activation='softmax', name='outcome_output')(l_o)

        model = Model(inputs=input_act, outputs=[output_l, output_o])
        print(model.summary())

        opt = Adam(lr=params["learning_rate"])

        model.compile(loss={'act_output':'categorical_crossentropy', 'outcome_output':'categorical_crossentropy'}, optimizer=opt, loss_weights=[params['gamma'], 1-params['gamma']] ,metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=20)
        lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto',
                                       min_delta=0.0001, cooldown=0, min_lr=0)

        # early_stopping = EarlyStopping(monitor='val_loss', patience=42)
        # #model_checkpoint = ModelCheckpoint('output_files/models/model_{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
        # lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

        history = model.fit(X_train, [Y_train, Z_train], epochs=500, batch_size=2**params['batch_size'], verbose=2, callbacks=[early_stopping, lr_reducer], validation_split =0.2 )

        scores = [history.history['val_loss'][epoch] for epoch in range(len(history.history['loss']))]
        #scores = [(params['gamma']*history.history['val_act_output_loss'][epoch])+(((1-params['gamma'])*history.history['val_outcome_output_loss'][epoch])) for epoch in range(len(history.history['loss']))]
        score = min(scores)
        global best_score, best_model
        if best_score > score:
                best_score = score
                best_model = model

        return {'loss': score, 'status': STATUS_OK}
        #model.save("model/generate_" + self.log_name + ".h5")

search_space = {'output_dim_embedding':scope.int(hp.loguniform('output_dim_embedding', np.log(10), np.log(150))),
                'shared_lstm_size': scope.int(hp.loguniform('shared_lstm_size', np.log(10), np.log(150))),
                'lstmA_size_1':  scope.int(hp.loguniform('lstmA_size_1', np.log(10), np.log(150))),
                'lstmO_size_1':  scope.int(hp.loguniform('lstmO_size_1', np.log(10), np.log(150))),
                'n_layers': hp.choice('n_layers', [
                {'n_layers': 1},
                {'n_layers': 2,
                    'lstmA_size_2_2': scope.int(hp.loguniform('lstmA_size_2_2', np.log(10), np.log(150))),
                    'lstmO_size_2_2': scope.int(hp.loguniform('lstmO_size_2_2', np.log(10), np.log(150))),
                 },
                {'n_layers': 3,
                    'lstmA_size_2_3': scope.int(hp.loguniform('lstmA_size_2_3', np.log(10), np.log(150))),
                    'lstmO_size_2_3': scope.int(hp.loguniform('lstmO_size_2_3', np.log(10), np.log(150))),
                    'lstmA_size_3_3': scope.int(hp.loguniform('lstmA_size_3_3', np.log(10), np.log(150))),
                    'lstmO_size_3_3': scope.int(hp.loguniform('lstmO_size_3_3', np.log(10), np.log(150)))}
                ]),
                'gamma': hp.uniform("gamma", 0.1,0.9),
                'dropout': hp.uniform("dropout", 0, 0.5),
                'batch_size': scope.int(hp.uniform('batch_size', 3, 6)),
                'learning_rate': hp.loguniform("learning_rate", np.log(0.00001), np.log(0.01))
                }
algorithm = tpe.suggest
best_score = np.inf
best_model = None

outfile = open('../Progetto-Tesi/data/log_files/' + log_name + '_opt.log', 'w')

trials = Trials()

best_params = fmin(
  fn=objective,
  space=search_space,
  algo=algorithm,
  max_evals=20,
  trials=trials)
print(len(trials))

best_params = space_eval(search_space,best_params)
print(best_params)

outfile.write("\nHyperopt trials")
outfile.write("\ntid,loss,output_dim_embedding,shared_lstm_size,lstmA_size_1,lstmO_size_1,n_layers,...,batch_size,learning_rate")
for trial in trials.trials:
    n_layers = trial['misc']['vals']['n_layers'][0]
    if(n_layers==1):
        outfile.write("\n%d,%f,%d, %d, %d, %d, %d, %f,%d,%f,%f" % (trial['tid'],
                                                trial['result']['loss'],
                                                trial['misc']['vals']['output_dim_embedding'][0],
                                                trial['misc']['vals']['shared_lstm_size'][0],
                                                trial['misc']['vals']['lstmA_size_1'][0],
                                                trial['misc']['vals']['lstmO_size_1'][0],
                                                trial['misc']['vals']['n_layers'][0],
                                                trial['misc']['vals']['dropout'][0],
                                                trial['misc']['vals']['batch_size'][0],
                                                trial['misc']['vals']['learning_rate'][0],
                                                trial['misc']['vals']['gamma'][0]
                                                ))
    elif(n_layers==2):
         outfile.write("\n%d,%f,%d, %d, %d, %d, %d, %f,%d,%f,%f" % (trial['tid'],
                                                trial['result']['loss'],
                                                trial['misc']['vals']['output_dim_embedding'][0],
                                                trial['misc']['vals']['shared_lstm_size'][0],
                                                trial['misc']['vals']['lstmA_size_1'][0],
                                                trial['misc']['vals']['lstmO_size_1'][0],
                                                trial['misc']['vals']['n_layers'][0],
                                                trial['misc']['vals']['dropout'][0],
                                                trial['misc']['vals']['batch_size'][0],
                                                trial['misc']['vals']['learning_rate'][0],
                                                trial['misc']['vals']['gamma'][0]
                                                ))
    elif(n_layers==3):
         outfile.write("\n%d,%f,%d, %d, %d, %d, %d, %f,%d,%f,%f" % (trial['tid'],
                                                trial['result']['loss'],
                                                trial['misc']['vals']['output_dim_embedding'][0],
                                                trial['misc']['vals']['shared_lstm_size'][0],
                                                trial['misc']['vals']['lstmA_size_1'][0],
                                                trial['misc']['vals']['lstmO_size_1'][0],
                                                trial['misc']['vals']['n_layers'][0],
                                                trial['misc']['vals']['dropout'][0],
                                                trial['misc']['vals']['batch_size'][0],
                                                trial['misc']['vals']['learning_rate'][0],
                                                trial['misc']['vals']['gamma'][0]
                                                ))

outfile.write("\n\nBest parameters:")
print(best_params, file=outfile)

best_model.save("model/generate_" + log_name + ".h5")

print('Evaluating final model...')
reportNA,reportO = manager.evaluate_model(X_test,Y_test,Z_test)
print(reportNA, file=outfile)
print(reportO, file=outfile)



