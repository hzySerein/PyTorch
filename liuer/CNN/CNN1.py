import torch


in_channel ,out_channel= 5,10
width ,height= 100,100
kernel_size = 3
batch_size = 2

input = torch.randn(batch_size,
                    in_channel,
                    width,
                    height)
convolution_layer = torch.nn.Conv2d(in_channel,
                                    out_channel,
                                    kernel_size,
                                    stride=1,
                                    padding=0)
output = convolution_layer(input)

print(input.shape)
print(output.shape)
print(convolution_layer.weight.shape)
