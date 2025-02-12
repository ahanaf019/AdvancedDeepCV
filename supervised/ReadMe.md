# Supervised Learning

## Description

Here, I trained a simple CNN model (CNNModel) using the STL10 dataset. The model had a ResNet50 encoder. Dropout was used in the model. It was trained using 1, 5, 10, 50 and 100 images per class. Training was done with and without Transfer Learning (TL). Testing was done using the Entire test set. I also limited training to only 50 epochs

## Results

| Model Name | TL  | No. Train Images | Accuracy | F1 Score | AUC    |
| ---------- | --- | ---------------: | -------- | -------- | ------ |
| CNNModel   | No  |                1 | 0.0890   | 0.0594   | 0.5626 |
| CNNModel   | No  |                5 | 0.0996   | 0.0261   | 0.4847 |
| CNNModel   | No  |               10 | 0.1000   | 0.0182   | 0.4895 |
| CNNModel   | No  |               50 | 0.1505   | 0.1035   | 0.5885 |
| CNNModel   | No  |              100 | 0.3076   | 0.2971   | 0.8026 |
| CNNModel   | Yes |                1 | 0.3263   | 0.3020   | 0.7136 |
| CNNModel   | Yes |                5 | 0.7365   | 0.7340   | 0.9437 |
| CNNModel   | Yes |               10 | 0.8739   | 0.8736   | 0.9865 |
| CNNModel   | Yes |               50 | 0.8729   | 0.8719   | 0.9908 |
| CNNModel   | Yes |              100 | 0.9238   | 0.9235   | 0.9959 |
