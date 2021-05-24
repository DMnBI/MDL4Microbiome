#!/usr/bin/env python3

import numpy as np
import pandas as pd
import keras, csv, argparse, os, sys
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import sklearn

#print(np.__version__)
#print(pd.__version__)
#print(keras.__version__)
#print(sklearn.__version__)
#print(csv.__version__)
#print(argparse.__version__)

def make_model(input_size, ty):

    # model for individual features
    if ty == 0:
        model = Sequential()
        model.add(Dense(200, input_dim=input_size, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # model for shared representation features
    elif ty == 1:
        model = Sequential()
        model.add(Dense(50, input_dim=input_size, activation='relu'))
        model.add(Dense(25, activation='relu'))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def train_individuals(df, y, key, epoch, tmp, logger):

    input_size = df.shape[1] 
    logger.write(" dataset shape: " + "("+ str(df.shape[0])+","+str(df.shape[1])+")\n")

    # LOOCV
    loo = LeaveOneOut()
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    cnt = 0
    for train_index, test_index in loo.split(df):
        print(key+"\t"+str(test_index[0]))
        print(train_index)
        x_train, x_test = df[train_index], df[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model = make_model(input_size, 0)
        model.fit(x_train, y_train, epochs=epoch, batch_size=1)
        model.summary()

        # the leading class label in alphabetical order 
        # is considered as positive state
        # get accuracy and (TP, TN, FP, FN) 
        a = np.argmax(y_test)
        b = np.argmax(model.predict(x_test))
        print(a,b)
        if a == b:
            print("correct")
            cnt += 1
        if a == 0 and b == 0:
            TP += 1
        elif a == 0 and b == 1:
            FN += 1
        elif a == 1 and b == 0:
            FP += 1
        elif a == 1 and b == 1:
            TN += 1
        else:
            print("ERROR!!")
            exit()
    

        inp = model.input
        outputs = [layer.output for layer in model.layers]
        functors = [K.function([inp], [out]) for out in outputs]
        layer_outs = [func([x_train]) for func in functors]

        # save train-data outputs
        with open("/".join([tmp, key])+"/train_"+str(test_index[0])+".csv", "w") as csvv:
            writer = csv.writer(csvv)
            for a in layer_outs[2][0]:
                writer.writerow(['{:.6f}'.format(x) for x in a])

        layer_outs = [func([x_test]) for func in functors]
        # save test-data outputs
        with open("/".join([tmp, key])+"/test_"+str(test_index[0])+".csv", "w") as csvv:
            writer = csv.writer(csvv)
            for a in layer_outs[2][0]:
                writer.writerow(['{:.6f}'.format(x) for x in a])

        keras.backend.clear_session()

    logger.write(" accuracy: \t"+str(cnt/df.shape[0])+"\n")
    logger.write("\t".join(["TP","FN","FP","TN"])+"\n")
    logger.write("\t".join([str(TP),str(FN),str(FP),str(TN)])+"\n\n")


def get_datasets(modals):

    f = open(modals)
    modal_list = f.read().splitlines()
    f.close()

    for modal in modal_list:
        if not os.path.isfile(modal):
            sys.exit(modal+"\tdoes not exist")

    return modal_list
        

def get_ylab(ylab_file):
    
    f = open(ylab_file)
    points = f.read().splitlines()
    f.close()

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(points)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded),1)
    ylab = onehot_encoder.fit_transform(integer_encoded)

    return ylab


def run_individuals(datasets, ylab, tmp, epoch, logger):
    
    individuals = []

    # multimodal - each modality
    for data in datasets:
        logger.write("########################################################\n")
        logger.write("\t"+data+"\n")
        logger.write("########################################################\n")
        
        dataset = pd.read_csv(data, sep=',', header=0)
        dataset = dataset.to_numpy().T

        key = data.split("/")[-1].split(".")[0]
        os.mkdir("/".join([tmp,key]))
        #os.mkdir("/".join([tmp]+key.split()[0:2]))
        
        train_individuals(dataset, ylab, key, epoch, tmp, logger)
        individuals.append("/".join([tmp, key]))

    return individuals, dataset.shape[0]


def run_shared(individuals, y, iter_num, epoch, logger):
 
    input_size = 50 * len(individuals)
    print(iter_num)

    # multimodal - final model
    logger.write("########################################################\n")
    logger.write('\tshared\n')
    logger.write("########################################################\n")
    logger.write(" dataset shape: " + "("+ str(iter_num)+","+str(input_size)+")\n")
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    cnt = 0
    for i in range(iter_num):
        print("shared\t"+str(i))

        #LOOCV
        indi_train = []
        indi_test = []
        for indi in individuals:
            tmp_train   = pd.read_csv(indi+"/train_"+str(i)+".csv", sep=',', header=None)
            tmp_test    = pd.read_csv(indi+"/test_"+str(i)+".csv", sep=',', header=None)
            indi_train.append(tmp_train)
            indi_test.append(tmp_test)

        train_set       = pd.concat(indi_train, ignore_index=True, axis=1)
        train_set       = train_set.to_numpy()
        test_set        = pd.concat(indi_test, ignore_index=True, axis=1)
        test_set        = test_set.to_numpy()
        print(train_set.shape, test_set.shape)

        train_y         = np.delete(y, i, axis=0)
        test_y          = y[i]
        print(train_y.shape, test_y.shape)

        model = make_model(input_size, 1)
        model.fit(train_set, train_y, epochs=epoch, batch_size=1)

        # the leading class label in alphabetical order 
        # is considered as positive state
        # get accuracy and (TP, TN, FP, FN) 
        a = np.argmax(test_y)
        b = np.argmax(model.predict(test_set))
        print(a,b)
        if a == b:
            print("correct")
            cnt += 1
        if a == 0 and b == 0:
            TP += 1
        elif a == 0 and b == 1:
            FN += 1
        elif a == 1 and b == 0:
            FP += 1
        elif a == 1 and b == 1:
            TN += 1
        else:
            print("ERROR!!")
            exit()
        keras.backend.clear_session()

    logger.write(" accuracy: " + str(cnt/iter_num)+'\n')
    logger.write("\t".join(["TP","FN","FP","TN"])+"\n")
    logger.write("\t".join([str(TP),str(FN),str(FP),str(TN)])+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
    """ Welcome to MDL4Microbiome.
    multi-modal deep learning model (aiming for microbiome samples)
    see README.md file for specific instructions.
    >> example commandline
    >> ./MDL4Microbiome.py -m examples/LC_list.txt -y examples/LC_ylabel.txt -t tmp/ -e1 30 -e2 10""",
    formatter_class = argparse.RawTextHelpFormatter, add_help=False)

    parser.add_argument("-m", "--modality",
                        dest="modals",
                        action="store",
                        required=True,
                        metavar='\b',
                        help="A file containing dataset file names (path from current directory). One file per line")
    parser.add_argument("-y", "--ylabel",
                        dest="ylab",
                        action="store",
                        required=True,
                        metavar='\b',
                        help="A file containing class label (binary) of data points. The leading class label in alphabetical order is considered as positive state")
    parser.add_argument("-t", "--tmp",
                        dest="tmp",
                        action="store",
                        required=True,
                        metavar='\b',
                        help="A directory for temporary files to be saved")
    parser.add_argument("-e1", "--epoch1",
                        dest="epoch1",
                        action="store",
                        required=True,
                        metavar='\b',
                        help="Number of epochs when training individual features")
    parser.add_argument("-e2", "--epoch2",
                        dest="epoch2",
                        action="store",
                        required=True,
                        metavar='\b',
                        help="Number of epochs when training shared representation features")
    parser.add_argument("-l", "--log",
                        dest="logger",
                        action="store",
                        required=True,
                        metavar='\b',
                        help="A name of a file for summarised results")
    parser.add_argument("-i", "--individuals",
                        dest="indi",
                        action="store_true",
                        help="A flag for \"individuals only\". Use when you want to see the classification results of individual features only.")
    parser.add_argument("-h", "--help", 
                        action="help",
                        help="show this help message and exit")
    
    args = parser.parse_args()

    datasets = get_datasets(args.modals)
    print(str(len(datasets)) + " datasets")
    ylab = get_ylab(args.ylab)

    logger = open(args.logger,"w")
    individuals, iter_num = run_individuals(datasets, ylab, args.tmp, int(args.epoch1), logger)
    if not args.indi:
        run_shared(individuals, ylab, iter_num, int(args.epoch2), logger)
    logger.close()
