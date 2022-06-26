from manager import Manager
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK

import numpy as np
import pandas as pd
from pathlib import Path
import datetime

from sklearn import preprocessing
from sklearn import metrics

from keras.models import load_model

from sklearn.metrics import confusion_matrix
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.sequence import pad_sequences


import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

log_name="finalThesisSUBDataset(09-06)_anon"
activity_name = "cat_item"
case_name = "request"
timestamp_name = "sys_updated_on"
outcome_name = "outcome"
delimiter = ','
win_size = 4

net_embedding = 1 #0 embedding layer, 1 word2vec
prediction = 0 #0 outcome, 1 completion time
time_view = 1 #0 no time view, 1 time view
time_type = "days"


train_w2v=1
word2vec_size = 179
w2vModelName = "generate_finalThesisDataset_anon_size179.h5"

nnModelName = "generateFINAL_finalThesisDataset_anon_type3_emb1_days.h5"
resultsFilename = "FINAL_finalThesisDataset_anon_1_TimeViewOutcomeOutput_days.log"





if(time_type == "seconds"):
    ending_name = "seconds_to_end"
    elapsed_time = "elapsed_seconds"
elif(time_type=="days"):
    ending_name = "days_to_end"
    elapsed_time = "elapsed_days"

print("Processing CSV...")

filename = str(Path('../Progetto-Tesi/data/sorted_csvs/' + log_name + ".csv").resolve())
data = pd.read_csv(filename, delimiter = delimiter)


data2 = data.filter([activity_name,case_name,timestamp_name,outcome_name,ending_name, "elapsed_seconds","elapsed_days"], axis=1)
data2[timestamp_name] = pd.to_datetime(data2[timestamp_name])
#data2[self.ending_name]
data_updated = data2.copy()

num_cases = len(data_updated[case_name].unique())
print("NUMERO TRACCE = " + str(num_cases))
num_events = data_updated.shape[0]
print("NUMERO EVENTI = " + str(num_events))

#print(data2.shape[0])
idx=0
for i, row in data2.iterrows():
    # print(i)
    # print(idx)
    if(i!=data2.shape[0]-1):
        if (row[case_name]!=data2.at[i+1,case_name]):
            idx+=1
            elapsedTimeValue = 0
            if(time_type=="seconds"):
                elapsedTimeValue = row[elapsed_time]+1
            elif(time_type =="days"):
                elapsedTimeValue = float(row["elapsed_seconds"]+1) /float(86400)

            line = pd.DataFrame({activity_name: row[outcome_name],
                                 case_name: row[case_name],
                                 timestamp_name: row[timestamp_name]+datetime.timedelta(seconds=1),
                                 outcome_name: row[outcome_name],
                                 elapsed_time: elapsedTimeValue,
                                 ending_name: 0
                                 }, index=[idx])
            data_updated = pd.concat([data_updated.iloc[:idx], line, data_updated.iloc[idx:]]).reset_index(drop=True)
    else: #ultima attività(ultimo caso), serve aggiungere l'evento outcome comunque
        #print("last")
        idx+=1
        elapsedTimeValue = 0
        if(time_type=="seconds"):
            elapsedTimeValue = row[elapsed_time]+1
        elif(time_type =="days"):
            elapsedTimeValue = float(row["elapsed_seconds"]+1) /float(86400)
        line = pd.DataFrame({activity_name: row[outcome_name],
                             case_name: row[case_name],
                             timestamp_name: row[timestamp_name]+datetime.timedelta(seconds=1),
                             outcome_name: row[outcome_name],
                             elapsed_time: elapsedTimeValue,
                             ending_name: 0
                             }, index=[idx])
        data_updated = pd.concat([data_updated.iloc[:idx],line])
    idx+=1

num_events = data_updated.shape[0]
print("NUMERO EVENTI DOPO UPDATE = " + str(num_events))

outsize_act = len(data[activity_name].unique())
#print(self.outsize_act)
outsize_out = len(data[outcome_name].unique())
#print(self.outsize_out)


data = data_updated.copy()
print("Done updating data")

print("Building dictionary...")
act_dictionary = {}
#inserisco per prime nel dizionario le label degli outcome
outcomes = data[outcome_name].unique()
for i in range(0,len(outcomes)):
        act_dictionary[outcomes[i]] = i+1
    # if(self.net_embedding==1):
    #     self.w2v_act.append(''.join(filter(str.isalnum, outcomes[i])))

#inserisco il resto delle attività nel dizionario
for i, event in data.iterrows():
    if event[activity_name] not in act_dictionary.keys():
        act_dictionary[event[activity_name]] = len(act_dictionary.keys()) + 1
        # if(self.net_embedding==1):
        #     # if('' not in self.w2v_act):
        #     #     self.w2v_act.append('')
        #     self.w2v_act.append(''.join(filter(str.isalnum, event[self.activity_name])))

print("Dictionary built:")
print(act_dictionary)

print("Building traces...")

traces = np.empty((len(data[case_name].unique()),), dtype=object)
traces[...]=[[] for _ in range(len(data[case_name].unique()))]

seconds_traces = np.zeros((len(data[case_name].unique()),), dtype=object)
seconds_traces[...]=[[] for _ in range(len(data[case_name].unique()))]

elapsed_traces = np.zeros((len(data[case_name].unique()),), dtype=object)
elapsed_traces[...]=[[] for _ in range(len(data[case_name].unique()))]

if(net_embedding==0):
    outcomes = range(1, len(data[outcome_name].unique())+1)
elif(net_embedding==1):
     name_outcomes = data[outcome_name].unique()
     outcomes = []
     for outcome in name_outcomes:
        outcomes.append(''.join(filter(str.isalnum, outcome)))
#print(outcomes)

traces_counter = 0
for i, event in data.iterrows():
    if(net_embedding==0):
        activity_coded = act_dictionary[event[activity_name]]
    elif(net_embedding==1):
        activity_coded = ''.join(filter(str.isalnum,event[activity_name]))
        #activity_coded = ''.join(filter(str.isalnum, activity_coded))
    #print(activity_coded)
    time_to_ending = event[ending_name]
    elapsed_time_val = event[elapsed_time]
    traces[traces_counter].append(activity_coded)

    elapsed_traces[traces_counter].append(elapsed_time_val)
    seconds_traces[traces_counter].append(time_to_ending)
    #print(self.traces[traces_counter])
    if(activity_coded in outcomes):
        traces_counter+=1

#converto ogni traccia in array piuttosto che liste
if(net_embedding==0):
    for i in range(0,len(traces)):
        elapsed_traces[i] = np.array(elapsed_traces[i])
        seconds_traces[i]= np.array(seconds_traces[i])
        traces[i] = np.array(traces[i])

# #print(list(self.act_dictionary.values()))
# if(self.net_embedding==0):
#     self.leA=self.leA.fit(list(self.act_dictionary.values()))
# elif(self.net_embedding==1):
#     self.leA.fit(self.getAnumAct(self.act_dictionary.keys()))
# #print(self.traces)
# self.traces_train, self.traces_test = train_test_split(self.traces, test_size=0.2, random_state=42, shuffle=False)
# # X_train,Y_train,Z_train = self.build_windows(self.traces_train,self.win_size)
# # print(X_train)
# # print(Y_train)
# # print(Z_train)
#
# # print("train")
# # print(self.traces_train)
# #
# # print("test")
# # print(self.traces_test)
#
#
#
# #self.getTraces(data)


print("Done: traces obtained, amount = ", len(traces))

print("Building windows...")

X_vec = [] #prefixes
el_time_vec = [] #view
Y_vec = [] #labels

print("Building windows...")
for i in range(0,len(traces)):
    trace= traces[i]
    sec_trace= seconds_traces[i]
    elapsed_trace = elapsed_traces[i]
    #print(sec_trace)
    #print(elapsed_trace)
    # print("TRACE:")
    # print(trace)
    k=0
    #print(i)
    j=1
    #while j < trace.size:
    while j < len(trace):
        #print(j)
        if(net_embedding==0):
            current_example = np.zeros(win_size)
        else:
            current_example= []
        time_current_example = np.zeros(win_size)

        values = trace[0:j] if j <= win_size else \
                 trace[j - win_size:j]
        time_values = elapsed_trace[0:j] if j <= win_size else \
                 elapsed_trace[j - win_size:j]
        if(prediction == 1 ):
            Y_vec.append(sec_trace[k])
        elif(prediction == 0):
            encoded_outcome = trace[len(trace)-1]
            Y_vec.append(encoded_outcome)

        #print(values)
        current_example[win_size - len(values):] = values
        time_current_example[win_size - len(values):] = time_values
        #print(time_current_example)
        #print(current_example)
        X_vec.append(current_example)
        el_time_vec.append(time_current_example)
        j += 1
        k+=1
    #print("NUOVA TRACCIA___________________")


if(net_embedding==0):
    X_vec = np.asarray(X_vec)
    el_time_vec = np.asarray(el_time_vec)
    Y_vec = np.asarray(Y_vec)

print("Done: windows built")

print("EVALUATION:")

if(net_embedding== 1):
    if(train_w2v==0):
        w2v_model = Word2Vec.load("models/w2v/"+w2vModelName)
    elif(train_w2v==1):
        w2v_model=  Word2Vec(vector_size=word2vec_size, seed=123,sg=0, min_count=1)
        #print(self.w2v_act)
        w2v_model.build_vocab(traces,min_count=1)

        total_examples = w2v_model.corpus_count
        # print("NUM corpus w2v")
        # print(total_examples)
        # addestro W2V
        w2v_model.train(traces, total_examples=total_examples, epochs=200)



    vocab = list(w2v_model.wv.index_to_key)
    w2v_embeddings = {}
    for word in vocab:
        w2v_embeddings[word] = w2v_model.wv.get_vector(word)

    X_vec = pad_sequences(X_vec, maxlen=4, padding='pre', dtype=object, value='_PAD_')

    X_vecNew = []
    for prefix in X_vec:
        list_temp_embed=[]
        #i = 0
        for act in prefix:
            embed_vector = w2v_embeddings.get(act)
            if embed_vector is not None: # word is in the vocabulary learned by the w2v model
                list_temp_embed.append(embed_vector)
                #print(len(prefix[i]))
            else:
                list_temp_embed.append(np.zeros(shape=(word2vec_size)))
                #print(len(prefix[i]))
        X_vecNew.append(list_temp_embed)
            #print(len(prefix))
            #i+=1
    X_vec=np.asarray(X_vecNew)
    el_time_vec = np.asarray(el_time_vec)

if(prediction == 0):
    leO = preprocessing.LabelEncoder()
    Y_vec = leO.fit_transform(Y_vec)

if(time_view == 0):
    input = X_vec
elif(time_view==1):
    input = [X_vec, el_time_vec]


model = load_model("models/"+nnModelName)
predictions = model.predict(input, batch_size=128, verbose = 0)
print(predictions)

outfile = open("../Progetto-Tesi/data/log_files/"+resultsFilename, "a")
if(prediction == 1):
    mae = metrics.mean_absolute_error(Y_vec, predictions)

    outfile.write("\nTime prediction metric:\n")
    print(mae, file=outfile)
elif(prediction == 0):
    rounded_out_prediction = np.argmax(predictions,axis=-1)
    cm_out = confusion_matrix(y_true= Y_vec, y_pred=rounded_out_prediction)
    cm_plot_labels_out = range(0,outsize_out)
    #print(cm_out)
    #self.plot_confusion_matrix(cm=cm_out, classes=cm_plot_labels_out, title='Confusion Matrix Outcome')
    reportO = metrics.classification_report(Y_vec, rounded_out_prediction, digits=3)

    outfile.write("\nOutcome metrics:\n")
    print(reportO, file=outfile)
    outfile.write("\nOutcome confusion matrix:\n")
    print(cm_out, file=outfile)
