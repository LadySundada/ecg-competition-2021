import wfdb
import os
import numpy as np
import tensorflow as tf
import sys
from utils import save_dict
import more_itertools as mit

model_path = 'ecg_ResNet.h5'

def getDataSet(sample_path, slice, label):
    sig, fields = wfdb.rdsamp(sample_path)
    sig = sig[:, 1]

    slice_tmp = np.zeros(SIZE)

    ann = wfdb.rdann(sample_path, 'atr')
    Rlocation = ann.sample
    Rclass = ann.symbol
    Rclass = [0] * len(Rclass)
    sample_descrip = fields['comments']
    ann_note = np.array(ann.aux_note)
    af_start = np.where((ann_note == '(AFIB') | (ann_note == '(AFL'))[0]
    af_end = np.where(ann_note == '(N')[0]

    end_points = []
    for j in range(len(af_start)):
        if sample_descrip == ['paroxysmal atrial fibrillation'] or sample_descrip == ['persistent atrial fibrillation']:
            start_end = [[af_start[j], af_end[j]]]
            end_points.extend(start_end)
        if end_points != [] and sample_descrip == ['persistent atrial fibrillation']:
            Rclass[end_points[j][0]:end_points[j][1] + 1] = np.ones(end_points[j][1] - end_points[j][0] + 1, int)
        if end_points != [] and sample_descrip == ['paroxysmal atrial fibrillation']:
            Rclass[end_points[j][0]:end_points[j][1] + 1] = np.ones(end_points[j][1] - end_points[j][0] + 1, int)

    for i in range(1, len(Rlocation)):
        slice_per_peak = sig[Rlocation[i-1] : Rlocation[i]]
        label_per_peak = Rclass[i]

        if len(slice_per_peak) <= SIZE:
            slice_tmp[0:len(slice_per_peak)] = slice_per_peak
            slice.append(slice_tmp)
            label.append(label_per_peak)
        else:
            slice_tmp = slice_per_peak[0:len(slice_tmp)]
            slice.append(slice_tmp)
            label.append(label_per_peak)

    return slice, label, Rlocation, Rclass

def Process(dataSet, labelSet):
    dataSet = np.array(dataSet).reshape(-1, SIZE)
    labelSet = np.array(labelSet).reshape(-1, 1)
    train_ds = np.hstack((dataSet, labelSet))

    X = train_ds[:, :SIZE].reshape(-1, SIZE, 1)
    Y = train_ds[:, SIZE]

    model = tf.keras.models.load_model(filepath=model_path)
    Y_pred = model.predict(X)
    Y_pred = np.argmax(Y_pred, axis=1)

    return Y, Y_pred

def merge(intervals):
    threshold = 50
    res = []
    intervals.sort()
    for i in intervals:
        if not res or i[0]-res[-1][1]>threshold:
            res.append(i)
        else:
            res[-1][1] = max(res[-1][1],i[1])
    return res

def data_clear(end_points):
    res = []
    for interval in end_points:
        if interval[-1] - interval[0] > 5:
            res.append(interval)

    return res

if __name__ == '__main__':
    DATA_PATH = sys.argv[1]
    RESULT_PATH = sys.argv[2]
    SIZE = 200
    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)

    test_set = open(os.path.join(DATA_PATH, 'RECORDS'), 'r').read().splitlines()
    new = open(os.path.join(DATA_PATH, 'REVISED_RECORDS'), 'r').read().splitlines()
    for data in new:
        num = test_set.index(data)
        test_set[num] = data

    for i, sample in enumerate(test_set):
        dataSet = []
        labelSet = []
        print(sample)
        sample_path = os.path.join(DATA_PATH, sample)
        dataSet, labelSet, Rloc, Rclass, = getDataSet(sample_path, dataSet, labelSet)
        Y, Y_pred = Process(dataSet, labelSet)

        pred = list(Y_pred)
        pred = [x for x in pred if x != '\n']
        index_pred = [i for i, x in enumerate(pred) if x == 1]

        end_points = []
        if len(pred) != 0:
            start_end_range = [list(group) for group in mit.consecutive_groups(index_pred)]
            for indices in start_end_range:
                for indices in start_end_range:
                    start_end = [[indices[0], indices[-1]]]
                    end_points.extend(start_end)

        end_points = merge(end_points)
        end_points = data_clear(end_points)

        if end_points != [] and end_points[0][0] == 0 and end_points[-1][1] > len(Rclass) - 50:
            end_points = [[0, len(Rclass)-1]]

        if len(end_points) != 0:
            for j in range(len(end_points)):
                end_points[j] = [Rloc[end_points[j][0]], Rloc[end_points[j][1]]-1]


        pred_dict = {'predict_endpoints': np.array(end_points).tolist()}
        save_dict(os.path.join(RESULT_PATH, sample + '.json'), pred_dict)