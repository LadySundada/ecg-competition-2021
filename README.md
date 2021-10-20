# Python code for the 4th China Physiological Signal Challenge 2021

## What's in this repository?

We implemented a ResNet DNN architecture to classify AF and PAF events from ECG lead signals.

## How do I run these scripts?

The pre-trained model was saved as 'ecg_ResNet.h5'. You can run this baseline method by installing the requirements

    pip install requirements.txt

and running 

    python entry_2021.py <data_path> <result_save_path>

where <data_path> is the folder path of the test set, <result_save_path> is the folder path of your detection results. 

## How do I run my code and save my results?

 The results will be saved as ‘.json’ files by record. The format is as {‘predict_endpoints’: [[s0, e0], [s1, e1], …, [sm-1, em-2]] }. The name of the result file should be the same as the corresponding record file.

After obtaining the test results, you can evaluate the scores of your method by running

    python score_2021.py <ans_path> <result_save_path>

where <ans_path> is the folder save the answers, which is the same path as <data_path> while the data and annotations are stored with 'wfdb' format. <result_save_path> is the folder path of your detection results.
