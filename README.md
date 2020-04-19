# Medical-Report-Classification
# 1. INTRODUCTION
Radiology reports contain rich information describing radiologist’s observations on the patient’s medical conditions in the associated medical images. Typically a report contains patient clinical history and the information revealed in the scan by the radiologist.
These reports can be classified as normal or abnormal based on the report content. It is difficult to analyse the Chest X-ray reports without consulting the radiologist. Machine learning algorithm can be trained with the labelled reports that will help in classifying the reports.

# 2. Dataset : 
A training dataset of 897 studies of chest X-Ray was collected from Australia LMI and KIMS for this project. The studies had X-ray scans along with the corresponding reports. While preparing the dataset for training each of the reports were manually segregated as normal and abnormal based on the terminologies used in reports. A total of 600 normal and 250 abnormal reports were used for training the model. The abnormalities present in this dataset includes consolidation, atelectasis, lung nodules, blunted costophrenic angle, opacity and pleural effusion. The dataset was split as 723 for training and 127 for validation of the model.(i.e. 85%  of total data for training and 15 % for validation) Further the model was tested with unlabelled reports to check the model’s performance.
B. Model Architecture
Here, logistic regression model is used for classifying reports as normal or abnormal.
 
### Collecting the reports for preparing training data set
Radiological X-Ray reports related to chest were collected from LMI and KIMS. Total of 850 reports were considered for training. These reports had findings related to lungs.

### Preparing the training data set
The reports that were considered for training (850) had findings related to lungs, heart and other parts of chest. Only the findings related to lungs were considered, other findings were removed since the model had to classify lungs as normal or abnormal. These data were then manually labelled as normal or abnormal.The first column is Label where normal reports are labelled as “1” and abnormal reports are labelled as “0”. The second column is the report id (from modality). Third column is the findings related to lungs.

### Pre Processing
The findings in the training data set has words that are redundant. This reduces the models accuracy. So we pre-process the training data set. Pre-processing involves removal of stop words, stemming and tokenizing.

#### Stop Words: 
The list of words that are not to be added in the features are called a stop words. A text file is prepared in which list of stop words were added. (e.g., is, was, the, and etc..,). While extracting the features these stop words are removed from the training data set.

##### Stemming:
is a technique to remove affixes from a word, ending up with the stem. For example, the stem of “observing” is “observe”. This reduces the redundancy of words in the features and improves the accuracy.

#### Algorithm used : 
One of the most common stemming algorithms is the Porter stemming algorithm by Martin Porter. It is designed to remove and replace well-known suffixes of English words.

#### White Space Tokenizer : 
Tokenization is the act of breaking up a sequence of strings into pieces such as words, keywords, phrases and other elements called tokens. Since we have to analyse the words in a sentence, we have used white space tokenizer to break the sentence into words. A white space tokenizer breaks sentence into words based on spaces. 


### Feature Extraction
Once the training data is passed through pre processing stage, we extract the features from the training data. These features are used to train the model. This gives the machine learning model a simpler and more focused view of the text. 
Steps involved in feature extraction:
Analyzer : We have used word analyser  while extracting features as we need the features of words. 
N-gram: An n-gram is a contiguous sequence of n items from a given sample of text. The module works by creating a dictionary of n-grams from a column of free text that you specify as input. 
e.g: n is number of words in a sequence that is to be considered.
We have used n-gram where n is 3, i.e Trigram. It considers sequence of three words.
The feature size was set to 150 as we got better result while validation.

# 3. Model 
The objective is to predict the chest X-Ray report as normal or abnormal. As it is a binary classification problem (0 or 1), we used logistic Regression  for classifying the reports. 
Logistic Regression is a classification algorithm. It is used to predict a binary outcome (1 / 0, Yes / No, True / False, Normal/Abnormal) given a set of independent variables. The model is trained with training data set along with the extracted features. 

# 4. Training
The logistic regression model was trained with 850 labelled reports. 600 normal reports and 250 abnormal reports were considered for training. 85% of the total data set i.e 723 labelled reports were used for  training and 15 % i.e 127 labelled reports were used as validation data set.

 # 5. Testing
  The model was tested with 60 un-labelled reports out of which 30 were normal and 30 were abnormal. 
  Out of the 30 normal reports the model predicted 24 normal as normal reports and 6 normal as abnormal reports. 
  Out of the 30 abnormal reports the model predicted 27 abnormal as abnormal reports and 3 abnormal as normal reports.
	
# 6. Evaluation Metrics
We used the following metrics to evaluate the performance of the model. 
Confusion Matrix[9] contains information about actual and predicted classifications done by a classification system. Performance of   such systems is commonly evaluated using data in the matrix. The following table shows the confusion matrix for a two class classifier.

# 7. RESULTS

The model was trained for 723  and validated for 127 reports. The classification report is  as shown in table . It was seen that model trained for 150 features gave an overall better result than the model trained for other features. 
###### Precision:0.97 	
###### Recall:0.97
###### F-1 Score: 0.97
The trained model  was tested with 30 normal and 30 abnormal radiology reports to get metrics values: testing accuracy, sensitivity, specificity and f1-score . 

##### Testing accuracy (%):85.2%	
##### Sensitivity: 0.92	
##### Specificity: 0.81
##### F-1 Score:0.861
	

# 8. CONCLUSIONS
We have presented a machine learning model that can classify reports into corresponding classes. This model can be used for other classification problems. Classification of radiology reports is a challenging task. In this paper, we have attempted to classify radiology reports as normal or abnormal using logistic regression. The performance of the model can be improved by reducing redundant data, reports having unique text and  other  parameters.
      

