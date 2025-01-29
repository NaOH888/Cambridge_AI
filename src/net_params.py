import torch
import torch.nn as nn

workers = 2

data_root = "../src"

data_fold = {"ORI" : "/dataset(flowers)",
             "DRM" : "/deepdream"}

<<<<<<< HEAD
learning_rate = 0.1

num_epoches = 10

batch_size = 32
=======
learning_rate = 0.001

num_epoches = 10

batch_size = 16
>>>>>>> 909c74031a5e818be9d304dad3d0f31736f198e2

input_image_size_x = 512
input_image_size_y = 512
input_image_depth = 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')