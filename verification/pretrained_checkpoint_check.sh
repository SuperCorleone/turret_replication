# 检查文件是否存在
ls -la data/pretrained/source_policies/

# 检查文件内容
python -c "
import torch
import os
path = 'data/pretrained/source_policies'
for f in os.listdir(path):
    if f.endswith('.pth'):
        print(f'{f}: {os.path.getsize(os.path.join(path, f))} bytes')
"