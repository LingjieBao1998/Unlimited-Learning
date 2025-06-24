## T5中文模型
https://github.com/shibing624/textgen/blob/main/docs/prompt-t5-base-chinese.md

## COT数据集
https://www.modelscope.cn/datasets/swift/Alpaca-CoT/summary
https://www.kaggle.com/datasets/konradb/chain-of-thought-collection/data?select=CoT_collection.json

## encoder-decoder model with COT
### kaist-ai/CoT-T5-3B or kaist-ai/CoT-T5-11B
```python
from transformers import T5Tokenizer, T5ForConditionalGeneration
tokenizer = T5Tokenizer.from_pretrained("kaist-ai/CoT-T5-3B")
model = T5ForConditionalGeneration.from_pretrained("kaist-ai/CoT-T5-3B")
input_text = "Read the Directions and try to pick among A,B,C,D.\n\nDirecitons: A good way to figure out the relationship in a given question is to make up a sentence that describes the relationship between the first two words. Then, try to use the same sentence to find out which of the answer choices completes the same relationship with the third word.\nQuestion: Odometer is to mileage as compass is to?\nOptions: (A) speed, (B) hiking, (C) needle, (D) direction.\nLet's think step by step.\n"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
outputs = model.generate(input_ids, max_length=500)
print(tokenizer.decode(outputs[0]))

```
