# WECModel
Attemptting to implement Shen, Yikang, et al. "Word Embedding Based Correlation Model for Question/Answer Matching." AAAI. 2017.  
Training data: Yahoo! L4 - Yahoo! Answers Manner Questions and L6 - Yahoo! Answers Comprehensive Questions and Answers part1.  
## Prerequisite
Python 3.6 & 2.7  
Java 1.8.0  
[Tensorflow](https://www.tensorflow.org/) (Works on Python 3.6)  
[word2vec](https://github.com/danielfrg/word2vec) (Works on Python 2.7)  
[THUTag](https://github.com/SongRb/THUTag) (Works on Linux)  
Note: I will try to replace the heavy THUTag tool dependency with a light Python script later.  
## Overview
![Workflow Graph](https://rawgithub.com/SongRb/WECModel/master/workflow.svg)  
## Manual  
### THUTag
`java -Xmx3G -jar tagsuggest.jar train.TrainWEC --input=../traindata/YahooPostL6-1-[time].dat --output=/mnt/hgfs/Code/thu-tag-workspace/trainWEC7 --config="dataType=KeywordPost;para=0.5;minwordfreq=10;mintagfreq=10;selfTrans=0.2;commonLimit=2"`  

## Performance
Currently 48% by using DAG@1 evaluation method without using negative label.  
