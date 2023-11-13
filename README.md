# FederatedLearningOutlierDetection

## How to run

TODO

## Note

You need a license of [gurobi](https://www.gurobi.com) to run the experiments.

## Baseline Results

My algorithms are BSVClassifier and OneClassSVM. Both were run 10 times as it can be seen in the code (commit 3bcddb0)

[Results Folder](./results/)

### AUC

|Classifer|pen-global     |breast-cancer  |letter         |pen-local      |annthyroid     |satellite      |shuttle        |aloi           |kd99           |
|---------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|
|oc-SVM   |0.9512 ± 0.0436|0.9721 ± 0.0102|0.5195 ± 0.0382|0.9543 ± 0.0130|0.5316 ± 0.0152|0.9549 ± 0.0021|0.9862 ± 0.0002|0.5319 ± 0.0021|0.9518 ± 0.0050|
|OneClassSVM|0.9494 ± 0.0391              |0.9793 ± 0.0012|0.6585 ± 0.0798                              |0.7144 ± 0.2516|0.4933 ± 0.0137|0.9137 ± 0.0284|0.9865 ± 0.0003|
|BSVClassifier (aka SVDD)|0.9996 ± 0.0009              |0.9955 ± 0.0092|0.8080 ± 0.1360                              |0.7445 ± 0.1723|0.5147 ± 0.0361|0.9395 ± 0.0054| |

### Average Precision

|Classifier|pen-global                   |breast-cancer|letter                                       |pen-local      |annthyroid     |satellite      |shuttle        |
|------|-----------------------------|-------------|---------------------------------------------|---------------|---------------|---------------|---------------|
|OneClassSVM|0.9931 ± 0.0059              |0.9995 ± 0.0000|0.9601 ± 0.0135                              |0.9992 ± 0.0010|0.9646 ± 0.0013|0.9980 ± 0.0007|0.9996 ± 0.0000|
|BSVClassifier (aka SVDD)|0.9999 ± 0.0001              |0.9999 ± 0.0003|0.9746 ± 0.0170                              |0.9994 ± 0.0007|0.9673 ± 0.0038|0.9989 ± 0.0001| |
