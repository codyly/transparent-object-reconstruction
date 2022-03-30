import os
import cv2
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
import numpy as np
import torch.optim as optim
from tqdm import tqdm
import sys
import dataset_generator_new as dgen
use_gpu = torch.cuda.is_available()


LIGHT_PATTERN_NUMS = 5 * 6 + 2 
LIGHT_PATTERN_HEIGHT = 128
LIGHT_PATTERN_WIDTH = 64
BATCH_SIZE = 4
OUTPUT_TRACE_NUMS = 5

class DAE(nn.Module):

    def __init__(self):
        super(DAE, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # Encoder
        self.conv1 = nn.Conv2d(1, LIGHT_PATTERN_NUMS, (1, LIGHT_PATTERN_HEIGHT * LIGHT_PATTERN_WIDTH))

        self.fc1 = nn.Linear(LIGHT_PATTERN_NUMS, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 512)
        self.fc5 = nn.Linear(512, 1024)
        self.fc6 = nn.Linear(1024, 2048)
        self.fc7 = nn.Linear(2048, 2048)
        # self.fc8 = nn.Linear(2048, 2048)
        self.fc9 = nn.Linear(2048, 4096)
        self.fc10 = nn.Linear(4096, 4096)
        self.fc11 = nn.Linear(4096, 8192)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.relu(self.conv1(x))
        # If the size is a square you can only specify a single number
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        # x = F.relu(self.fc8(x))
        x = F.relu(self.fc9(x))
        x = F.relu(self.fc10(x))

        x = self.fc11(x)
        return x

    def predict(self, x):
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        # x = F.relu(self.fc8(x))
        x = F.relu(self.fc9(x))
        x = F.relu(self.fc10(x))

        x = self.fc11(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class STDataset(Dataset):
    def __init__(self, root_dir='./ds-prac', csv_file='dataset.csv', transform=None):
        csv_file_path = os.path.join(root_dir, csv_file)
        self.ds_info = pd.read_csv(csv_file_path)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.ds_info)

    def update(self):
        dgen.gen()

    def __getitem__(self, idx):
        json_file_path = os.path.join(self.root_dir, self.ds_info.iloc[idx, 0])
        json_str = None
        with open(json_file_path, 'r', encoding='utf-8') as f:
            json_str = json.load(f)
        sample = {'input': np.array(json_str['input']).reshape((1,1,-1)).astype(np.float32) / 255,
                  'output': np.array(json_str['output']).astype(np.float32) / 255}
        return sample

def test_and_visualize(model, test_dataLoder):
    with torch.no_grad():
        test_cnt = 1
        for data in test_dataLoder:
            images = data['input']
            outputs = data['output']
            test_outputs = model(images)
            sample_image = np.array(images)[2].reshape([LIGHT_PATTERN_HEIGHT, LIGHT_PATTERN_WIDTH])
            sample_traces = np.array(outputs)[2].reshape([LIGHT_PATTERN_HEIGHT, LIGHT_PATTERN_WIDTH])
            sample_tests = np.array(test_outputs)[2].reshape([LIGHT_PATTERN_HEIGHT, LIGHT_PATTERN_WIDTH])
            final_img = sample_image
            final_img = np.hstack([final_img, sample_traces])
            final_img = np.hstack([final_img, sample_tests])
            cv2.imshow('output', final_img)
            cv2.waitKey(0)
            break

class My_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y):
        return torch.mean(torch.pow((x - y) * (y * 3 + 1), 2))


if __name__ == '__main__':
    print(use_gpu)
    if use_gpu:
        device = torch.device("cuda:0")
    net = DAE()
    net.load_state_dict(torch.load('./model/parameter_new_new_pro_pro_pro_pro.pkl'))
    ds = STDataset()
    ds_loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    ds_val = STDataset(root_dir='./ds-val')
    val_loader = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    if len(sys.argv) > 1 and sys.argv[1] in 'test':
      test_and_visualize(net, ds_loader)

    elif len(sys.argv) > 1 and sys.argv[1] in 'predict':
      jstr = None
      filename = sys.argv[2]
      with open(filename + '.json', 'r', encoding='utf-8') as f:
        jstr = json.load(f)
      pred = np.array(jstr['data']).reshape([1,1,1,32])
      pred = torch.Tensor(pred)
      # print(pred)
      cnt = 0
      for param in net.parameters():
        if cnt == 1:
            pred += param.data
            break
        cnt += 1
      output = net.predict(pred).reshape([128,64])
      output = output.detach().numpy()
      cv2.imwrite('./predict/'+filename+'.png', output * 255)
      # cv2.imshow('output', output)
      # cv2.waitKey()
    
    elif len(sys.argv) > 1 and sys.argv[1] in 'train':

      # criterion = nn.MSELoss()
      criterion = My_loss()
      optimizer = optim.Adam(net.parameters(), lr=0.001)
      StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.65)

      epochs = 100
      checks = 50
      for epoch in tqdm(range(epochs)):  # loop over the dataset multiple times

          running_loss = 0.0
          for i, data in enumerate(ds_loader):
              # get the inputs
              inputs = data['input']
              labels = data['output']

              # print(inputs.shape)
              # zero the parameter gradients
              optimizer.zero_grad()

              # forward + backward + optimize
              outputs = net(inputs)
              loss = criterion(outputs, labels)
              loss.backward()
              optimizer.step()

              ds.update()

              for p in net.parameters():
                p.data.clamp_(min=0, max=255)
                break

              # print statistics
              running_loss += loss.item()
              if i % checks == checks - 1:    # print every 2000 mini-batches
                  val_loss = 0
                  for j, val_data in enumerate(val_loader):
                    vinputs = val_data['input']
                    vlabels = val_data['output']
                    voutputs = net(vinputs)
                    vloss = criterion(voutputs, vlabels)
                    val_loss += vloss.item()
                    break
                  print('\n[%d, %5d] loss: %.4f, val_loss: %.4f' %
                        (epoch + 1, i + 1, running_loss / checks, val_loss))
                  running_loss = 0.0
          if epoch % 20 == 19:
              torch.save(net.state_dict(), './model/parameter_new_new_pro_pro_pro_pro.pkl')
      print('Finished Training')