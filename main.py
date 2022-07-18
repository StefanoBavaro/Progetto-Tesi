import os

from manager import Manager
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK

import numpy as np
import pandas as pd
import datetime
from pathlib import Path
import itertools
import datetime

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

import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

log_name="Production_Sorted" #event log name
activity_name = "Activity" #name of the activity column
case_name = "Case ID" #name of the case id column
timestamp_name = "Complete Timestamp" #name of the timestamp column
outcome_name = "label" #name of the outcome column
delimiter = ',' #delimiter of the event log
elapsedTimeCol_name = "nullElapsed"  #name of the elapsed time column (put "nullElapsed" if not used)
remainingTimeCol_name = "nullRemaining" #name of the remaining time column (put "nullRemaining" if not used)

win_size = 4 #size of the sliding windows
net_out = 0 #0 = double output(outcome and next activity) ; 1 = nextActivity ; 2= outcome ; 3 = completion time
net_in = 0 #0 = no time view; 1 = time view
net_embedding = 0 #0 = layer embedding, 1 = word2vec
time_unit = -1 #0 = seconds, 1 = days, -1 = no time used

manager = Manager(log_name, activity_name, case_name, timestamp_name, outcome_name,win_size, net_out, net_embedding, delimiter, time_unit, net_in, elapsedTimeCol_name, remainingTimeCol_name)
manager.gen_internal_csv()
manager.csv_to_data()

if(net_out==0):
    search_space = {'output_dim_embedding':scope.int(hp.loguniform('output_dim_embedding', np.log(10), np.log(150))),
                    'word2vec_size': hp.uniformint('word2vec_size',32,1024),
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

elif(net_out==1 or net_out ==3): #layers lstm used for next activity prediction are used for completion time prediction as well
    search_space = {'output_dim_embedding':scope.int(hp.loguniform('output_dim_embedding', np.log(10), np.log(150))),
                    'word2vec_size': hp.uniformint('word2vec_size',32,1024),
                    'shared_lstm_size': scope.int(hp.loguniform('shared_lstm_size', np.log(10), np.log(150))),
                    'lstmA_size_1':  scope.int(hp.loguniform('lstmA_size_1', np.log(10), np.log(150))),
                    #'lstmO_size_1':  scope.int(hp.loguniform('lstmO_size_1', np.log(10), np.log(150))),
                    'n_layers': hp.choice('n_layers', [
                    {'n_layers': 1},
                    {'n_layers': 2,
                        'lstmA_size_2_2': scope.int(hp.loguniform('lstmA_size_2_2', np.log(10), np.log(150))),
                        #'lstmO_size_2_2': scope.int(hp.loguniform('lstmO_size_2_2', np.log(10), np.log(150))),
                     },
                    {'n_layers': 3,
                        'lstmA_size_2_3': scope.int(hp.loguniform('lstmA_size_2_3', np.log(10), np.log(150))),
                        #'lstmO_size_2_3': scope.int(hp.loguniform('lstmO_size_2_3', np.log(10), np.log(150))),
                        'lstmA_size_3_3': scope.int(hp.loguniform('lstmA_size_3_3', np.log(10), np.log(150))),
                        #'lstmO_size_3_3': scope.int(hp.loguniform('lstmO_size_3_3', np.log(10), np.log(150)))
                     }
                    ]),
                    'dropout': hp.uniform("dropout", 0, 0.5),
                    'batch_size': scope.int(hp.uniform('batch_size', 3, 6)),
                    'learning_rate': hp.loguniform("learning_rate", np.log(0.00001), np.log(0.01))
                    }

elif(net_out==2):
     search_space = {'output_dim_embedding':scope.int(hp.loguniform('output_dim_embedding', np.log(10), np.log(150))),
                     'shared_lstm_size': scope.int(hp.loguniform('shared_lstm_size', np.log(10), np.log(150))),
                     'word2vec_size': hp.uniformint('word2vec_size',32,1024),
                    #'lstmA_size_1':  scope.int(hp.loguniform('lstmA_size_1', np.log(10), np.log(150))),
                    'lstmO_size_1':  scope.int(hp.loguniform('lstmO_size_1', np.log(10), np.log(150))),
                    'n_layers': hp.choice('n_layers', [
                    {'n_layers': 1},
                    {'n_layers': 2,
                        #'lstmA_size_2_2': scope.int(hp.loguniform('lstmA_size_2_2', np.log(10), np.log(150))),
                        'lstmO_size_2_2': scope.int(hp.loguniform('lstmO_size_2_2', np.log(10), np.log(150))),
                     },
                    {'n_layers': 3,
                        #'lstmA_size_2_3': scope.int(hp.loguniform('lstmA_size_2_3', np.log(10), np.log(150))),
                        'lstmO_size_2_3': scope.int(hp.loguniform('lstmO_size_2_3', np.log(10), np.log(150))),
                        #'lstmA_size_3_3': scope.int(hp.loguniform('lstmA_size_3_3', np.log(10), np.log(150))),
                        'lstmO_size_3_3': scope.int(hp.loguniform('lstmO_size_3_3', np.log(10), np.log(150)))
                     }
                    ]),
                    'dropout': hp.uniform("dropout", 0, 0.5),
                    'batch_size': scope.int(hp.uniform('batch_size', 3, 6)),
                    'learning_rate': hp.loguniform("learning_rate", np.log(0.00001), np.log(0.01))
                    }

commonFileName = log_name+'_Emb'+str(net_embedding)+ '_Task'+str(net_out)+'_TimeView' + str(net_in)+ '_TimeUnit'+str(time_unit)


algorithm = tpe.suggest
best_params, trials = manager.fmin(
      fn=manager.nn,
      space=search_space,
      algo=algorithm,
      max_evals=2,
      filename =commonFileName)

print(len(trials))

best_params = space_eval(search_space,best_params)
print(best_params)


outfile = open('../Progetto-Tesi/data/log_files/' + commonFileName + '.log', 'w')

outfile.write("\nHyperopt trials:")
outfile.write("\ntid,loss,time,params used")
totaltime =0
for trial in trials.trials:
    #print(trial)
    outfile.write("\n%d, %f, %f, %s" % (trial['tid'],
                            trial['result']['loss'],
                            trial['result']['time'],
                            trial['misc']['vals']))
    totaltime = totaltime + float(trial['result']['time'])

totaltime = str(datetime.timedelta(seconds=totaltime))

outfile.write("\nTotal time: %s" % totaltime)
outfile.write("\n\nBest parameters:")
print(best_params, file=outfile)

manager.best_model.save("models/generate_"+commonFileName+".h5")
print('Evaluating final models...')
model= manager.best_model
if(net_out==0):
    reportNA,cmNA,reportO,cmO = manager.evaluate_model(model,best_params['word2vec_size'])
    outfile.write("\nNext activity metrics:\n")
    print(reportNA, file=outfile)
    outfile.write("\nNext activity confusion matrix:\n")
    print(cmNA, file=outfile)
    outfile.write("\nOutcome metrics:\n")
    print(reportO, file=outfile)
    outfile.write("\nOutcome confusion matrix:\n")
    print(cmO, file=outfile)
elif(net_out==1):
    reportNA,cmNA = manager.evaluate_model(model,best_params['word2vec_size'])
    outfile.write("\nNext activity metrics:\n")
    print(reportNA, file=outfile)
    outfile.write("\nNext activity confusion matrix:\n")
    print(cmNA, file=outfile)
elif(net_out==2):
    reportO,cmO = manager.evaluate_model(model,best_params['word2vec_size'])
    outfile.write("\nOutcome metrics:\n")
    print(reportO, file=outfile)
    outfile.write("\nOutcome confusion matrix:\n")
    print(cmO, file=outfile)
elif(net_out==3):
    mae = manager.evaluate_model(model, best_params['word2vec_size'])
    outfile.write("\nTime prediction metrics:\n")
    print(mae, file=outfile)

