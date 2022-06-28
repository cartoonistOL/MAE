#TODO
# 将时间str转换为tensor
import torch
class to_tensor():
    def __call__(self, timelist : list) -> torch.Tensor:
        timels_tensor = []
        for time in timelist:
            timels_tensor.append([float(time)])
        return torch.as_tensor(timels_tensor)

# timelist = ['01', '01', '11', '12', '11', '11', '10', '11', '11', '09']
# print(to(timelist))