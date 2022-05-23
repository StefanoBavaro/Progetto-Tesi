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


log_name="BPIC11_f1_Sorted"
activity_name = "Activity code"
case_name = "Case ID"
timestamp_name = "time:timestamp"
outcome_name = "label"
win_size = 4
net_out = 1 #0 = double output ; 1 = nextActivity net ; 2= outcome net
net_embedding = 0 #0 = embedding, 1 = word2vec
delimiter = ';'


manager = Manager(log_name, activity_name, case_name, timestamp_name, outcome_name,win_size, net_out, net_embedding, delimiter)
manager.gen_internal_csv()
manager.csv_to_data()

algorithm = tpe.suggest

# try:  # try to load an already saved trials object, and increase the max
#     trials = pickle.load(open("my_model.hyperopt", "rb"))x
#     print("Found saved Trials! Loading...")
#     max_trials = len(trials.trials) + trials_step
#     print("Rerunning from {} trials to {} (+{}) trials".format(len(trials.trials), max_trials, trials_step))
# except:  # create a new trials object and start searching
#     trials = Trials()


trials = Trials()

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
    outfile = open('../Progetto-Tesi/data/log_files/' + log_name +'_'+str(net_embedding) +'_doubleOutput.log', 'w')
elif(net_out==1):
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
                    'win_size':4,
                    'dropout': hp.uniform("dropout", 0, 0.5),
                    'batch_size': scope.int(hp.uniform('batch_size', 3, 6)),
                    'learning_rate': hp.loguniform("learning_rate", np.log(0.00001), np.log(0.01))
                    }
    outfile = open('../Progetto-Tesi/data/log_files/' + log_name+'_'+str(net_embedding) + '_singleActOutput.log', 'w')
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
                    'win_size': 4,
                    #'gamma': hp.uniform("gamma", 0.1,0.9),
                    'dropout': hp.uniform("dropout", 0, 0.5),
                    'batch_size': scope.int(hp.uniform('batch_size', 3, 6)),
                    'learning_rate': hp.loguniform("learning_rate", np.log(0.00001), np.log(0.01))
                    }
     outfile = open('../Progetto-Tesi/data/log_files/' + log_name +'_'+str(net_embedding)+ '_singleOutOutput.log', 'w')

# try:
#     os.makedirs(Path('../Progetto-Tesi/models/hpTrials/'+ log_name +'_'+ str(net_embedding)+ '_'+str(net_out)))
# except FileExistsError:
#     print("Directory already exists \n")
# trialsFilename = '../Progetto-Tesi/models/hpTrials/'+ log_name +'_'+ str(net_embedding)+ '_'+str(net_out)+ '/'+log_name+'_'+str(net_embedding)+ '_'+str(net_out)

trialsFilename = '../Progetto-Tesi/models/hpTrials/'+log_name+'_'+str(net_embedding)+ '_'+str(net_out)


best_params, trials = manager.fmin(
      fn=manager.nn,
      space=search_space,
      algo=algorithm,
      max_evals=3,
      filename =trialsFilename)




# best_params = fmin(
#       fn=manager.nn,
#       space=search_space,
#       algo=algorithm,
#       max_evals=20,
#       trials=trials)
print(len(trials))

best_params = space_eval(search_space,best_params)
print(best_params)

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

manager.best_model.save("models/generate_" + log_name + "_type"+str(net_out)+"_emb"+str(net_embedding) + ".h5")

print('Evaluating final models...')
model= load_model("models/generate_" + log_name + "_type"+str(net_out)+"_emb"+str(net_embedding) + ".h5")
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




