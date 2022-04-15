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
#example_size = 4

manager = Manager(log_name, activity_name, case_name, timestamp_name, outcome_name)
manager.gen_internal_csv()
manager.csv_to_data()


#train_traces,test_traces = manager.csv_to_data()
#X_train, X_test, Y_trainBefore, Y_test, Z_trainBefore, Z_test = manager.csv_to_data()
# manager.build_neural_network_model(X_train,Y_train,Z_train)
# manager.evaluate_model(X_test,Y_test,Z_test)


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
                'win_size': hp.choice("win_size", [4, 8, 16, 32]),
                'gamma': hp.uniform("gamma", 0.1,0.9),
                'dropout': hp.uniform("dropout", 0, 0.5),
                'batch_size': scope.int(hp.uniform('batch_size', 3, 6)),
                'learning_rate': hp.loguniform("learning_rate", np.log(0.00001), np.log(0.01))
                }
algorithm = tpe.suggest
# best_score = np.inf
# best_model = None

outfile = open('../Progetto-Tesi/data/log_files/' + log_name + '_opt.log', 'w')

trials = Trials()

best_params = fmin(
  fn=manager.objective,
  space=search_space,
  algo=algorithm,
  max_evals=3,
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

manager.best_model.save("model/generate_" + log_name + ".h5")

print('Evaluating final model...')
reportNA,reportO = manager.evaluate_model(best_params['win_size'])
print(reportNA, file=outfile)
print(reportO, file=outfile)



