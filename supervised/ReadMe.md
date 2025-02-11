# Supervised Learning

## Description

Here, I trained a simple CNN model (CNNModel) using the STL10 dataset. The model had a ResNet50 encoder. Dropout was used in the model. It was trained using 1, 5, 10, 50 and 100 images per class. Training was done with and without Transfer Learning (TL). Testing was done using the Entire test set. I also limited training to only 50 epochs

## Results

| Model Name | TL  | No. Train Images | Accuracy | F1 Score | AUC    |
| ---------- | --- | ---------------: | -------- | -------- | ------ |
| CNNModel   | No  |                1 | 0.1035   | 0.0307   | 0.5293 |
| CNNModel   | No  |                5 | 0.0930   | 0.0284   | 0.5257 |
| CNNModel   | No  |               10 | 0.1001   | 0.0184   | 0.5378 |
| CNNModel   | No  |               50 | 0.1841   | 0.1504   | 0.6782 |
| CNNModel   | No  |              100 | 0.1664   | 0.1235   | 0.6204 |
| CNNModel   | Yes |                1 | 0.3265   | 0.7284   | 0.2467 |
| CNNModel   | Yes |                5 | 0.6505   | 0.6020   | 0.9505 |
| CNNModel   | Yes |               10 | 0.7059   | 0.6763   | 0.9515 |
| CNNModel   | Yes |               50 | 0.9091   | 0.9936   | 0.9091 |
| CNNModel   | Yes |              100 | 0.9219   | 0.9219   | 0.9951 |
