# Detection-of-Adversarial-Devices-using-ML

## 1. Dataset: 
[CICIoT dataset 2023](https://www.unb.ca/cic/datasets/iotdataset-2023.html)   
 - __originnal traffic:__ in .pcap files, split the data using TCPDump into multiple files(2GB each), Mergecap can be used to merge the data.   
 - __formatted data:__ extracted from .pcap files into .csv files. All 169 .csv files refer to a combined and shuffled dataset including all attacks. The attacks are identified by the ‘label’ feature.


## 2. __Preprocessing:__
- __Data Aggregation:__ `data_aggregation.py` The original size is over 15GB, change the data type(from float64 to ...), keep the 22 important features, aggregate the whole dataset into one .pkl file(CICIoT2023.pkl).   
-  __Grouping:__  `data_group.py`  
  - sample data from CICIoT2023.pkl, choose a percentage with the balanced class
  - Define Attack Groups:  attack, category and subcategory labels
    - 2 classes: Benign or Malicious  
    - 8 classes: Benign, DoS, DDoS, Recon, Mirai, Spoofing, Web, BruteForce
    - 34 classes: subgroups of the above 8 classes
   
## 3. __Classifiers:__  
### 3.1 Random Forest
- Default Random Forest: n_estimators=100    
- Evaluation Metrics: Accuracy, Precision, Recall, F1-score
### 3.2 DNN
### 3.3 NLP methods maybe.
