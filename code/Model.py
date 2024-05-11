
import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping

from keras.layers import Input, Conv1D, AveragePooling1D, GlobalAveragePooling1D, \
    Dropout, Dense, Activation, Concatenate, Multiply, GlobalMaxPooling1D, Add, GRU, \
    LSTM, Bidirectional, Conv2D, AveragePooling2D, BatchNormalization, Flatten, GlobalAveragePooling2D, \
    GlobalMaxPooling2D, Reshape, Permute, multiply, Lambda, add, subtract, MaxPooling2D, LeakyReLU, ELU
from keras.regularizers import l1, l2
from keras.optimizer_v2.adam import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from keras.models import Model, load_model
import tensorflow as tf
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold  
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from keras.layers import Layer
from keras import initializers

import warnings

warnings.filterwarnings("ignore")



def read_fasta(fasta_file_name):
    seqs = []
    seqs_num = 0
    file = open(fasta_file_name)

    for line in file.readlines():
        if line.strip() == '':
            continue

        if line.startswith('>'):
            seqs_num = seqs_num + 1
            continue
        else:
            seq = line.strip()

            result1 = 'N' in seq
            result2 = 'n' in seq
            if result1 == False and result2 == False:
                seqs.append(seq)
    return seqs



def to_one_hot(seqs):
    base_dict = {
        'a': 0, 'c': 1, 'g': 2, 't': 3,
        'A': 0, 'C': 1, 'G': 2, 'T': 3
    }

    one_hot_4_seqs = []
    for seq in seqs:

        one_hot_matrix = np.zeros([4, len(seq)], dtype=float)
        index = 0
        for seq_base in seq:
            one_hot_matrix[base_dict[seq_base], index] = 1
            index = index + 1

        one_hot_4_seqs.append(one_hot_matrix)
    return one_hot_4_seqs


def to_properties_density_code(seqs):
    properties_code_dict = {
        'A': [1, 1, 1], 'C': [0, 1, 0], 'G': [1, 0, 0], 'T': [0, 0, 1],
        'a': [1, 1, 1], 'c': [0, 1, 0], 'g': [1, 0, 0], 't': [0, 0, 1]
    }
    properties_code = []
    for seq in seqs:
        properties_matrix = np.zeros([4, len(seq)], dtype=float)
        A_num = 0
        C_num = 0
        G_num = 0
        T_num = 0
        All_num = 0
        for seq_base in seq:
            if seq_base == "A":
                All_num += 1
                A_num += 1
                Density = A_num / All_num
                properties_matrix[:, All_num - 1] = properties_code_dict[seq_base] + [Density]
            if seq_base == "C":
                All_num += 1
                C_num += 1
                Density = C_num / All_num
                properties_matrix[:, All_num - 1] = properties_code_dict[seq_base] + [Density]
            if seq_base == "G":
                All_num += 1
                G_num += 1
                Density = G_num / All_num
                properties_matrix[:, All_num - 1] = properties_code_dict[seq_base] + [Density]
            if seq_base == "T":
                All_num += 1
                T_num += 1
                Density = T_num / All_num
                properties_matrix[:, All_num - 1] = properties_code_dict[seq_base] + [Density]
        properties_code.append(properties_matrix)
    return properties_code



def show_performance(y_true, y_pred):
    
    TP, FP, FN, TN = 0, 0, 0, 0

    for i in range(len(y_true)):
        if y_true[i] == 1:
            if y_pred[i] > 0.5:
                TP += 1
            else:
                FN += 1
        if y_true[i] == 0:
            if y_pred[i] > 0.5:
                FP += 1
            else:
                TN += 1

    
    Sn = TP / (TP + FN + 1e-06)
    
    Sp = TN / (FP + TN + 1e-06)
    
    Acc = (TP + TN) / len(y_true)
    
    MCC = ((TP * TN) - (FP * FN)) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + 1e-06)

    return Sn, Sp, Acc, MCC

def performance_mean(performance):
    print('Sn = %.4f ± %.4f' % (np.mean(performance[:, 0]), np.std(performance[:, 0])))
    print('Sp = %.4f ± %.4f' % (np.mean(performance[:, 1]), np.std(performance[:, 1])))
    print('Acc = %.4f ± %.4f' % (np.mean(performance[:, 2]), np.std(performance[:, 2])))
    print('Mcc = %.4f ± %.4f' % (np.mean(performance[:, 3]), np.std(performance[:, 3])))
    print('Auc = %.4f ± %.4f' % (np.mean(performance[:, 4]), np.std(performance[:, 4])))




if __name__ == '__main__':

    np.random.seed(0)
    tf.random.set_seed(1)  # for reproducibility


    train_pos_seqs = np.array(read_fasta('../datasets/train_positive_data.fasta'))
    train_neg_seqs = np.array(read_fasta('../datasets/train_negative_data.fasta'))

    train_seqs = np.concatenate((train_pos_seqs, train_neg_seqs), axis=0)


    train_onehot = np.array(to_one_hot(train_seqs)).astype(np.float32)
    train_properties_code = np.array(to_properties_density_code(train_seqs)).astype(np.float32)

    train = np.concatenate((train_onehot, train_properties_code), axis=1)



    train_label = np.array([1] * 55800 + [0] * 658858).astype(np.float32)
    train_label = to_categorical(train_label, num_classes=2)


    test_pos_seqs = np.array(read_fasta('../datasets/test_positive_data.fasta'))
    test_neg_seqs = np.array(read_fasta('../datasets/test_negative_data.fasta'))

    test_seqs = np.concatenate((test_pos_seqs, test_neg_seqs), axis=0)


    test_onehot = np.array(to_one_hot(test_seqs)).astype(np.float32)
    test_properties_code = np.array(to_properties_density_code(test_seqs)).astype(np.float32)

    test = np.concatenate((test_onehot, test_properties_code), axis=1)


    test_label = np.array([1] * 13950 + [0] * 164713).astype(np.float32)
    test_label = to_categorical(test_label, num_classes=2)



    n = 5
    k_fold = KFold(n_splits=n, shuffle=True, random_state=42)


    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    all_performance = []
    for fold_count,(train_index, val_index) in enumerate(k_fold.split(train)):
        print('*' * 30 + ' fold ' + str(fold_count+1) + ' ' + '*' * 30)
        tra, val = train[train_index], train[val_index]
        tra_label, val_label = train_label[train_index], train_label[val_index]



        model1 = load_model('../models/5mC_model_' + str(fold_count+1) + '_1.h5', custom_objects={'AttLayer': AttLayer})
        model2 = load_model('../models/5mC_model_' + str(fold_count+1) + '_2.h5', custom_objects={'AttLayer': AttLayer})
        model3 = load_model('../models/5mC_model_' + str(fold_count+1) + '_3.h5', custom_objects={'AttLayer': AttLayer})


        val_score1 = model1.predict(val)
        val_score2 = model2.predict(val)
        val_score3 = model3.predict(val)
        all_score = val_score1 + val_score2 + val_score3
        val_score = all_score / 3


        Sn, Sp, Acc, MCC = show_performance(val_label[:, 1], val_score[:, 1])
        AUC = roc_auc_score(val_label[:, 1], val_score[:, 1])
        print('Sn = %f, Sp = %f, Acc = %f, MCC = %f, AUC = %f' % (Sn, Sp, Acc, MCC, AUC))

        performance = [Sn, Sp, Acc, MCC, AUC]
        all_performance.append(performance)

        '''Mapping the ROC'''
        fpr, tpr, thresholds = roc_curve(val_label[:, 1], val_score[:, 1], pos_label=1)

        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        plt.plot(fpr, tpr, label='ROC fold {} (AUC={:.4f})'.format(str(fold_count + 1), AUC))

        fold_count += 1
    all_performance = np.array(all_performance)
    print('5 fold result:',all_performance)
    performance_mean = performance_mean(all_performance)

    '''Mapping the ROC'''
    plt.plot([0, 1], [0, 1], '--', color='red')
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0

    mean_auc = np.mean(np.array(all_performance)[:, 4])

    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC=%0.4f)' % (mean_auc), lw=2, alpha=.8)

    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig('../images/5fold_ROC_Curve.jpg', dpi=1200, bbox_inches='tight')
    plt.legend(loc='lower right')
    plt.show()



    model1 = load_model('../models/5mC_model1.h5', custom_objects={'AttLayer': AttLayer})
    model2 = load_model('../models/5mC_model2.h5', custom_objects={'AttLayer': AttLayer})
    model3 = load_model('../models/5mC_model3.h5', custom_objects={'AttLayer': AttLayer})


    test_score1 = model1.predict(test)
    test_score2 = model2.predict(test)
    test_score3 = model3.predict(test)
    all_score = test_score1 + test_score2 + test_score3
    test_score = all_score / 3


    Sn, Sp, Acc, MCC = show_performance(test_label[:, 1], test_score[:, 1])
    AUC = roc_auc_score(test_label[:, 1], test_score[:, 1])

    print('-----------------------------------------------test---------------------------------------')
    print('Sn = %f, Sp = %f, Acc = %f, MCC = %f, AUC = %f' % (Sn, Sp, Acc, MCC, AUC))

    '''Mapping the test ROC'''
    plt.plot([0, 1], [0, 1], '--', color='red')
    test_fpr, test_tpr, thresholds = roc_curve(test_label[:, 1], test_score[:, 1], pos_label=1)

    plt.plot(test_fpr, test_tpr, color='b', label=r'DGA_5mC (AUC=%0.4f)' % (AUC), lw=2, alpha=.8)

    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig('../images/test_ROC_Curve.jpg', dpi=1200, bbox_inches='tight')
    plt.legend(loc='lower right')
    plt.show()
