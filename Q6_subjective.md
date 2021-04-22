## Nueral Net

### Classification
- For classifications of digits problem the following network has been defined:
<br />
NN_Layer((X_train.shape[0],X_train.shape[1]), (X_train.shape[0],20),'sigmoid'), <br />
NN_Layer((X_train.shape[0],20),(X_train.shape[0],n_classes),'softmax')

The following accuracy was achieved on using 3 fold cross validation: <br />

100%|██████████| 300/300 [00:11<00:00, 26.84it/s] <br />
100%|██████████| 300/300 [00:11<00:00, 27.02it/s] <br />
100%|██████████| 300/300 [00:11<00:00, 26.99it/s] <br />
Accuracies for 3 fold model are  [0.9515859766277128, 0.9415692821368948, 0.9131886477462438]

### Regression
- For regression of boston housing problem the following network has been defined:
<br />
NN_Layer((X_train.shape[0],X_train.shape[1]), (X_train.shape[0],20),'relu'),<br />
NN_Layer((X_train.shape[0],20),(X_train.shape[0],1),'relu')

The following rmse value was achieved on using 3 fold cross validation: <br />

100%|██████████| 300/300 [00:05<00:00, 51.22it/s] <br />
100%|██████████| 300/300 [00:05<00:00, 51.19it/s] <br />
100%|██████████| 300/300 [00:05<00:00, 51.12it/s] <br />
RMSE for 3 fold model are  [2.5218012750073453, 2.4725650701693174, 2.4923906103031412]