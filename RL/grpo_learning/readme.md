## 参考
> `grpo_encoder_decoder_summarization`：https://gist.github.com/jogonba2/9bee8bb154a292b24850f1483daa6b71 —— <a href="./grpo_encoder_decoder_summarization.py">grpo_encoder_decoder_summarization.py</a>
> 
> `Image_Caption_GRPO`:https://github.com/liangxu-one/ms-models/blob/image_caption_grpo/research/arxiv_papers/Image_Caption_GRPO/train_with_grpo.py (有些包不好安装,没有实现,不过代码的细节翔实)
> 
> `SFT+GRPO 混合训练`:https://www.kaggle.com/code/stpeteishii/medical-qa-bart-w-grpo-fine-tuning（没有KL 散度限制）——<a href="./Medical_QA_Bart_w_GRPO_Fine-Tuning.py">Medical_QA_Bart_w_GRPO_Fine-Tuning.py</a>

> 本文主要围绕第一个教程进行展开

## 报错
### nltk导入失败失败
```markdown
LookupError: 
**********************************************************************
  Resource punkt_tab not found.
  Please use the NLTK Downloader to obtain the resource:

  >>> import nltk
  >>> nltk.download('punkt_tab')
  
  For more information see: https://www.nltk.org/data.html

  Attempted to load tokenizers/punkt_tab/english/

  Searched in:
    - '/home/lingjiebao/anaconda3/envs/py39/nltk_data'
    - '/home/lingjiebao/nltk_data'
    - '/home/lingjiebao/anaconda3/envs/py39/nltk_data'
    - '/home/lingjiebao/anaconda3/envs/py39/share/nltk_data'
    - '/home/lingjiebao/anaconda3/envs/py39/lib/nltk_data'
    - '/usr/share/nltk_data'
    - '/usr/local/share/nltk_data'
    - '/usr/lib/nltk_data'
    - '/usr/local/lib/nltk_data'
**********************************************************************
```

#### 解决方案
手动下载在<a href="http://www.nltk.org/nltk_data/">http://www.nltk.org/nltk_data/</a>下载`punkt_tab.zip`, 然后解压到`下述任意目录/tokernizer/`文件夹下

```markdown
- '/home/lingjiebao/anaconda3/envs/py39/nltk_data'
- '/home/lingjiebao/nltk_data'
- '/home/lingjiebao/anaconda3/envs/py39/nltk_data'
- '/home/lingjiebao/anaconda3/envs/py39/share/nltk_data'
- '/home/lingjiebao/anaconda3/envs/py39/lib/nltk_data'
- '/usr/share/nltk_data'
- '/usr/local/share/nltk_data'
- '/usr/lib/nltk_data'
- '/usr/local/lib/nltk_data'
```

## 实现细节&重要参数（待添加）
### 运行
```bash
python grpo_encoder_decoder_summarization.py
```



## 建议
先进行`SFT`微调，再进行`GRPO`微调
<img src="assets/grpo_advice.png" alert="grpo_advice"></img>
> ref:https://rabiloo.com/blog/fine-tuning-a-reasoning-model-with-grpo-for-passport-data-extraction