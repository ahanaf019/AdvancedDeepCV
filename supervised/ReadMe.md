# Supervised Learning

## Description

Here, I trained a simple CNN model (CNNModel) using the STL10 dataset. The model had a ResNet50 encoder. Dropout was used in the model. It was trained using 10, 50 and 100 images per class. Training was done with and without Transfer Learning (TL). Testing was done using the Entire test set.


## Results

| Model Name | TL  | No. Train Images | Accuracy | F1 Score | AUC |
| ---------- | --- | ---------------: | -------- | -------- | --- |
| CNNModel   | No  |               10 |          |          |     |
| CNNModel   | No  |               50 |          |          |     |
| CNNModel   | No  |              100 |          |          |     |
| CNNModel   | Yes |               10 |          |          |     |
| CNNModel   | Yes |               50 |          |          |     |
| CNNModel   | Yes |              100 |          |          |     |
