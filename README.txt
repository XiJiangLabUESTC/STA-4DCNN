e-mail address: jiadong.yan@mail.mcgill.ca

tensorflow 1.10 on GTX 1080 Ti

STA-4DCNN_spatial.py
(1) inputs:
all inputs are defined in function load_data()
"train_path" is the path of the preprocessed brain data
"path1" is the path of the labels

(2) outputs:
"result.txt" file to record the training loss and testing loss
and the space.mat file is the result of the modeled targeted RSN spatial pattern

STA-4DCNN_temporal.py
(1) inputs:
all inputs are defined in function load_data()
"train_path" is the path of the preprocessed brain data
"path2" is the path of the labels
"spatial_p" is the modeled spatial patterns via spatial network which is also the input of the temporal network

(2) outputs:
"result.txt" file to record the training loss and testing loss
and the time.mat file is the result of the modeled targeted RSN temporal pattern
