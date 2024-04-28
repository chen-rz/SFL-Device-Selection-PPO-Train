import utils

model = utils.AlexNet()

# param_size = 0
# for param in model.parameters():
#     param_size += param.nelement() * param.element_size()
# buffer_size = 0
# for buffer in model.buffers():
#     buffer_size += buffer.nelement() * buffer.element_size()

# size_all_bits = (param_size + buffer_size) * 8
# print(size_all_bits)

from pthflops import count_ops
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

inputTensor = torch.rand(1, 3, 224, 224).to(device) # AlexNet

model.to(device)

count_ops(model, inputTensor)
