## 参考文件
> https://gist.github.com/jogonba2/9bee8bb154a292b24850f1483daa6b71


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

## 实现细节&重要参数


## 建议
先进行`SFT`微调，再进行`GRPO`微调
> ref:https://rabiloo.com/blog/fine-tuning-a-reasoning-model-with-grpo-for-passport-data-extraction