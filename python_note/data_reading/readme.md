

## parquet
```python
import pandas as pd

# 读取 Parquet 文件
file_path = 'aligned_chunk_243.parquet'
data = pd.read_parquet(file_path)
print(data.columns)
```