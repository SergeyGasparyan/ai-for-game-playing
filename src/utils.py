import cv2
import torch
import torch.nn as nn
import numpy as np


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def preprocessing(image):
	image_data = cv2.cvtColor(cv2.resize(image, (84, 84)), cv2.COLOR_BGR2GRAY)
	image_data[image_data > 0] = 255
	image_data = np.reshape(image_data,(84, 84, 1))
	image_tensor = image_data.transpose(2, 0, 1)
	image_tensor = image_tensor.astype(np.float32)
	image_tensor = torch.from_numpy(image_tensor).to(device)

	return image_tensor


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.uniform(m.weight, -0.01, 0.01)
        m.bias.data.fill_(0.01)
