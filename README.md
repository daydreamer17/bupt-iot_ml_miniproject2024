# bupt-iot_ml_miniproject2024
-*project background*:mini-project considers the problem of predicting whether a narrated story is true or not. Specifically, you will build a machine learning model that takes as an input an audio recording of **3-5 minutes** of duration and predicts whether the story being narrated is **true or not**. 

-*dataset we use*:A total of 100 samples consisting of a complete audio recording, a *Language* attribute and a *Story Type* attribute have been made available for you to build your machine learning model. The audio recordings can be downloaded from:

https://github.com/CBU5201Datasets/Deception

A CSV file recording the *Language* attribute and *Story Type* of each audio file can be downloaded from:

https://github.com/CBU5201Datasets/Deception/blob/main/CBU0521DD_stories_attributes.csv

-*model we use*:In this study, we aim to develop a CNN deep learning model that combines audio features and language types to accurately distinguish true from false audio content (e.g., identifying deceptive statements).

-*dataprocess.py*:we pre-process the dataset we download by this file's code

-*model*:we define a two-dense layers CNN model in this file

-*train.py*:it defines the way we train the code

-*CBU5201_miniproject.ipynb*:for more details of our approach,please see this jupyter file
