# TROCR
ref:https://qiita.com/relu/items/c027c486758525c0b6b9
ref:https://colab.research.google.com/drive/14MfFkhgPS63RJcP7rpBOK6OII_y34jx_?usp=sharing


## dataset

### 测试集
```bash
wget https://github.com/AakashKumarNain/CaptchaCracker/raw/master/captcha_images_v2.zip 
unzip -q captcha_images_v2.zip
mkdir -p IAM && mv captcha_images_v2 IAM/image
cd IAM/image && for fname in `ls *.png`; do echo -e "$fname\t${fname:0:5}" >> ../gt_test.txt; done ## 制作标签
```
> 文件名就是他的标签


## evaluation
```
pip install sentencepiece
```

## run
```bash
python main.py
```