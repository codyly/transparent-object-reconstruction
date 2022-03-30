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
from tqdm import trange
import sys
import dataset_generator_new as dgen
import math

use_gpu = torch.cuda.is_available()


LIGHT_PATTERN_NUMS = 1
LIGHT_PATTERN_HEIGHT = 128
LIGHT_PATTERN_WIDTH = 64
BATCH_SIZE = 4
OUTPUT_TRACE_NUMS = 5

class DAE(nn.Module):

    def __init__(self):
        super(DAE, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # Encoder
        self.convs = nn.ModuleList()
        for i in range(LIGHT_PATTERN_WIDTH):
          self.convs.append(nn.Conv2d(1, LIGHT_PATTERN_NUMS, (1, LIGHT_PATTERN_HEIGHT)))
        self.fc1 = nn.Linear(LIGHT_PATTERN_WIDTH * LIGHT_PATTERN_NUMS, 128)
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

        b = F.relu(self.convs[0](x[:,:,:, 0:LIGHT_PATTERN_HEIGHT]))

        for i in range(1, LIGHT_PATTERN_WIDTH):
          b = torch.cat((b, F.relu(self.convs[i](x[:,:,:, i*LIGHT_PATTERN_HEIGHT:(i+1)*LIGHT_PATTERN_HEIGHT]))),3)

        # If the size is a square you can only specify a single number
        x = b.view(-1, self.num_flat_features(b))
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

    def check(self, x):
        b = F.relu(self.convs[0](x[:,:,:, 0:LIGHT_PATTERN_HEIGHT]))

        for i in range(1, LIGHT_PATTERN_WIDTH):
          b = torch.cat((b, F.relu(self.convs[i](x[:,:,:, i*LIGHT_PATTERN_HEIGHT:(i+1)*LIGHT_PATTERN_HEIGHT]))),3)

        # If the size is a square you can only specify a single number
        x = b.view(-1, self.num_flat_features(b))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

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

class STDataset(Dataset):
    def __init__(self, root_dir='./ds-prac', csv_file='dataset.csv', transform=None):
        csv_file_path = os.path.join(root_dir, csv_file)
        self.ds_info = pd.read_csv(csv_file_path)
        self.root_dir = root_dir
        self.transform = transform
        self.csv_file = csv_file

    def __len__(self):
        return len(self.ds_info)

    def update(self):
        dgen.gen()

    def __getitem__(self, idx):
        sample = None
        if self.root_dir in './ds-st':
          img = cv2.imread("./ds-st/scatterTrace.png", cv2.IMREAD_GRAYSCALE)
          label = img.reshape([LIGHT_PATTERN_WIDTH * LIGHT_PATTERN_HEIGHT])
          img = img.reshape([LIGHT_PATTERN_HEIGHT, LIGHT_PATTERN_WIDTH]).T
          img = img.reshape([LIGHT_PATTERN_WIDTH * LIGHT_PATTERN_HEIGHT])
          sample = {'input': img.reshape((1,1,-1)).astype(np.float32) / 255,
                  'output': np.array(label).astype(np.float32) / 255}
        else:
          json_file_path = os.path.join(self.root_dir, self.ds_info.iloc[idx, 0])
          json_str = None
          with open(json_file_path, 'r', encoding='utf-8') as f:
              json_str = json.load(f)
          img = np.array(json_str['input']).reshape([LIGHT_PATTERN_HEIGHT, LIGHT_PATTERN_WIDTH]).T
          img = img.reshape([LIGHT_PATTERN_WIDTH * LIGHT_PATTERN_HEIGHT])
          sample = {'input': img.reshape((1,1,-1)).astype(np.float32) / 255,
                    'output': np.array(json_str['output']).astype(np.float32) / 255}
        return sample

def test_and_visualize(model, test_dataLoder, testId = 1):
    final_img = None
    with torch.no_grad():
        test_cnt = 1
        for data in test_dataLoder:
            if test_cnt < 2:
              test_cnt += 1
              continue
            images = data['input']
            outputs = data['output']
            test_outputs = model(images)
            sample_image = np.array(images)[3].reshape([LIGHT_PATTERN_WIDTH, LIGHT_PATTERN_HEIGHT]).T.reshape([LIGHT_PATTERN_HEIGHT, LIGHT_PATTERN_WIDTH])
            sample_traces = np.array(outputs)[3].reshape([LIGHT_PATTERN_HEIGHT, LIGHT_PATTERN_WIDTH])
            sample_tests = np.array(test_outputs)[3].reshape([LIGHT_PATTERN_HEIGHT, LIGHT_PATTERN_WIDTH])
            final_img = sample_image
            final_img = np.hstack([final_img, sample_traces])
            final_img = np.hstack([final_img, sample_tests])
            cv2.imwrite('./test/logs/testImage-epoch-{:03d}.png'.format(testId), final_img*255)
            # cv2.waitKey(0)
            break

def test_real_case(model, testId = 1, tag = 'st'):
    final_img = None
    BATCH_SIZE = 1

    ds_st = STDataset(root_dir='./ds-'+tag)
    st_loader = DataLoader(ds_st, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    with torch.no_grad():
        test_cnt = 1
        for data in st_loader:
            images = data['input']
            outputs = data['output']
            test_outputs = model(images)
            sample_image = np.array(images)[0].reshape([LIGHT_PATTERN_WIDTH, LIGHT_PATTERN_HEIGHT]).T.reshape([LIGHT_PATTERN_HEIGHT, LIGHT_PATTERN_WIDTH])
            sample_traces = np.array(outputs)[0].reshape([LIGHT_PATTERN_HEIGHT, LIGHT_PATTERN_WIDTH])
            sample_tests = np.array(test_outputs)[0].reshape([LIGHT_PATTERN_HEIGHT, LIGHT_PATTERN_WIDTH])
            final_img = sample_image
            final_img = np.hstack([final_img, sample_traces])
            final_img = np.hstack([final_img, sample_tests])
            cv2.imwrite('./test/'+tag+'/testImage_{:03d}.png'.format(test_cnt), final_img*255)
            test_cnt += 1
            print(model.check(images).detach().numpy().reshape([64,]).tolist())
            # cv2.imwrite('./test/realST/test.png', sample_tests*255)
            # cv2.waitKey(0)
            # break

class My_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y):
        return torch.mean(torch.pow((x - y) * (y * 4 + 1), 2))


if __name__ == '__main__':
    print(use_gpu)
    if use_gpu:
        device = torch.device("cuda:0")
    net = DAE()
    net.load_state_dict(torch.load('./model/parameter_latest_4_1_1.pkl'))
    net.eval()
    # net = torch.load('./model/parameter_practice_model.pt')
    # net.eval()
    ds = STDataset()
    ds_loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    ds_val = STDataset(root_dir='./ds-val')
    val_loader = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    if len(sys.argv) > 1 and sys.argv[1] in 'test':
      # test_and_visualize(net, ds_loader)
      # for name, param in net.named_parameters():
      #   print(param.shape)
      test_real_case(net)

    elif len(sys.argv) > 1 and sys.argv[1] in 'pattern':

      cnt = 0
      patterns = []
      for name, param in net.named_parameters():
        # print(name, param.shape)
        if cnt < 128 and cnt % 2 == 0:
          a = param.data.detach().numpy().reshape([128,])
          patterns.append(a)
        cnt += 1

      patterns = np.array(patterns).reshape([64,128])

      cv2.imshow('pattern', patterns.T)
      cv2.waitKey()
      sys.exit(0)      
      patterns = np.flip(patterns, 0)
      patterns = np.flip(patterns, 1)
      cv2.imshow('pattern', patterns.T)
      cv2.waitKey()
      patterns = patterns.reshape([1,64,128])
      # print(np.max(patterns))
      jstr = {"data":patterns.tolist()}
      with open('D:\\Study\\graphics\\graduation\\work\\simulation\\jsonGen\\practice.json', 'w+', encoding='utf-8') as f:
        json.dump(jstr, f)
      with open('D:\\Study\\graphics\\graduation\\work\\reconstruction\\practice.json', 'w+', encoding='utf-8') as f:
        json.dump(jstr, f)


    elif len(sys.argv) > 1 and sys.argv[1] in 'predict':
      jstr = None
      filename = sys.argv[2]
      with open('prac_' + filename + '.json', 'r', encoding='utf-8') as f:
        jstr = json.load(f)
      pred = np.array(jstr['data']).reshape([64,])
      # print(pred)
      for i in range(32):
        t = pred[i]
        pred[i] = pred[63-i]
        pred[63-i] = t
      pred = pred.reshape([1,64])
      pred = torch.Tensor(pred)
      # print(pred)
      cnt = 0
      for param in net.parameters():
        if cnt < 128 and cnt % 2 == 1:
          pred[0][cnt//2] += param.data[0]
          pred[0][cnt//2] = max(0, pred[0][cnt//2])
        cnt += 1

      # pred.reshape([1,1,64])
      print(pred.reshape([64,]).detach().numpy().tolist())
      output = net.predict(pred).reshape([128,64])
      output = output.detach().numpy()
      cv2.imwrite('./predict/prac_'+filename+'.png', output * 255)
      # cv2.imshow('output', output)
      # cv2.waitKey()

    elif len(sys.argv) > 1 and sys.argv[1] in 'edge':
      jstr = None
      filename = sys.argv[2]
      folder   = sys.argv[3]
      with open('prac_' + filename + '.json', 'r', encoding='utf-8') as f:
        jstr = json.load(f)

      preds = np.array(jstr['data'])

      for idx in trange(preds.shape[0]):
        pred = preds[idx].reshape([64,])
        # print(pred)
        for i in range(32):
          t = pred[i]
          pred[i] = pred[63-i]
          pred[63-i] = t
        pred = pred.reshape([1,64])
        pred = torch.Tensor(pred)
        # print(pred)
        cnt = 0
        for param in net.parameters():
          if cnt < 128 and cnt % 2 == 1:
            pred[0][cnt//2] += param.data[0]
            pred[0][cnt//2] = max(0, pred[0][cnt//2])
          cnt += 1

        # pred.reshape([1,1,64])
        # print(pred.reshape([64,]).detach().numpy().tolist())
        output = net.predict(pred).reshape([128,64])
        output = output.detach().numpy()
        cv2.imwrite('./predict/' + folder + '/prac_'+filename +'_' +'{:03d}'.format(idx)+'.png', output * 255)

    elif len(sys.argv) > 1 and sys.argv[1] in 'surface':
      jstr = None
      filename = sys.argv[2]
      folder   = sys.argv[3]
      with open('prac_' + filename + '.json', 'r', encoding='utf-8') as f:
        jstr = json.load(f)

      preds = np.array(jstr['data'])
      print(preds.shape)
      ccnt = -1

      for idx in trange(preds.shape[0]):
        for idy in range(preds.shape[1]):
          ccnt += 1
          pred = preds[idx][idy].reshape([64,])
          # print(pred)
          for i in range(32):
            t = pred[i]
            pred[i] = pred[63-i]
            pred[63-i] = t
          pred = pred.reshape([1,64])
          pred = torch.Tensor(pred)
          # print(pred)
          cnt = 0
          for param in net.parameters():
            if cnt < 128 and cnt % 2 == 1:
              pred[0][cnt//2] += param.data[0]
              pred[0][cnt//2] = max(0, pred[0][cnt//2])
            cnt += 1

          # pred.reshape([1,1,64])
          # print(pred.reshape([64,]).detach().numpy().tolist())
          output = net.predict(pred).reshape([128,64])
          output = output.detach().numpy()
          img = np.zeros([130, 64]).astype(np.float32)
          img[1:129,:] = output * 255
          img[np.where(img < 50)] = 0
          # def edge(g1, g2):
          #   if math.fabs(g1-g2) > 20:
          #     return True
          #   return False
          # for c in range(64):
          #   flag = False
          #   start = end = 0
          #   start_color = 0
          #   for r in range(1,128):
          #     if edge(img[r,c], img[r+1,c]):
          #       if flag == False and img[r+1,c] > img[r,c]:          # positive
          #         start = r + 1
          #         flag = True
          #         start_color = img[r,c]
          #       if flag == True and img[r+1,c] < img[r,c] and edge(start_color, img[r+1,c]) == False:           # negtive
          #         end = r
          #         flag = False
          #         if end - start + 1 > 2:
          #           interval = int((end-start + 1) / 2)
          #           for i in range(interval-2):
          #             img[end - i, c] = img[end + 1, c]
          #             img[start + i, c] = img[start - 1, c]
          #           start = end = 0
          output = img[1:129, :]

          cv2.imwrite('./predict/' + folder + '/prac_'+filename +'_' +'{:05d}'.format(ccnt)+'.png', output)

    elif len(sys.argv) > 1 and sys.argv[1] in 'train':

      # criterion = nn.MSELoss()
      criterion = My_loss()
      optimizer = optim.Adam(net.parameters(), lr=0.0001)
      scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.65)

      epochs = 100
      checks = 20
      checkpoint = 10
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

              cnt = 0
              for p in net.parameters():
                if cnt < 128 and cnt%2==0:
                  p.data.clamp_(min=0, max=255)
                cnt +=1
              
              ds.update()

              # print statistics
              running_loss += loss.item()
              if i % checks == checks - 1:    # print every 2000 mini-batches
                  val_loss = 0
                  cnt = 0
                  for j, val_data in enumerate(val_loader):
                    vinputs = val_data['input']
                    vlabels = val_data['output']
                    voutputs = net(vinputs)
                    vloss = criterion(voutputs, vlabels)
                    val_loss += vloss.item()
                    cnt += 1
                    # if cnt == 10:
                      # break
                  print('\n[%d, %5d] loss: %.4f, val_loss: %.4f' %
                        (epoch + 1, i + 1, running_loss / checks, val_loss / cnt))
                  running_loss = 0.0
          scheduler.step()
          lr = scheduler.get_last_lr()
          print('\nLearning Rate: %.4f' % lr[0])
          test_and_visualize(net, val_loader, epoch)
          if epoch % checkpoint == checkpoint - 1:
              torch.save(net.state_dict(), './model/parameter_latest_4_1_1.pkl')
      print('Finished Training')  