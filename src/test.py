import torch
print(torch.cuda.is_available())        # True 表示可以用 GPU
print(torch.cuda.device_count())        # GPU 数量
print(torch.cuda.get_device_name(0))    # GPU 型号（如果有的话）