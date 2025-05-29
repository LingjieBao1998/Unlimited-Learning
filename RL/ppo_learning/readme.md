
## data
`training.1600000.processed.noemoticon.csv`:https://www.kaggle.com/datasets/ferno2/training1600000processednoemoticoncsv


## 问题
### vader_lexicon下载失败
```
>>> import nltk
>>> nltk.download("vader_lexicon")
[nltk_data] Error loading vader_lexicon: <urlopen error [Errno 111]
[nltk_data]     Connection refused>
False
```
解决方案
```bash
pip install vaderSentiment
```
[SentimentIntensityAnalyzer_learning.ipynb](./SentimentIntensityAnalyzer_learning.ipynb)



## 参考
https://medium.com/geekculture/positivity-unleashed-ppo-and-generative-text-models-575fad035a9e


## TODO
* 解决模型微调的问题
* 检查PPO训练的细节
* 增加评测的metric
