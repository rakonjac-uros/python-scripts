from model import EAST
import torch
model = torch.load("pths/vgg16_bn-6c64b313.pth")
device = "cuda:0"
model = EAST('pths/vgg16_bn-6c64b313.pth').to(device)
model.load_state_dict(torch.load('pths/east_vgg16.pth', map_location=device))
model.eval()
dummy_input = torch.randn(1, 3, 2496, 3744)
torch.onnx.export(model, (dmmy_input,), 'east_vgg16_img.onnx')
