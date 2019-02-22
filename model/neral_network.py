# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib
import random
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch.utils.data as Data
import torch.nn.init as Init
import sys
import time


BATCH_SIZE = 50
LR = 0.01
EPOCH = 1000
decay_rate = 0.8
CUDA=1
inputElementCount = 23

class MyDataset(Data.Dataset):
    def __init__(self, data_dir):
        length_mean = 886324.4203868032
        length_std = 487978.5584822787
        bitrate_mean = 1846883.860341297
        bitrate_std = 972278.4525951932
        qoe_mean = 0.9099758992946531
        qoe_std = 0.7717136726511163
        buffers_mean = 25612.523949943115
        buffers_std = 5177.9566989276955
        throughputs_mean = 3608433.83011489
        throughputs_std = 6767026.597908898
        ssim_mean = 0.9623334166821902
        ssim_std = 0.04038052851566692
        bitrate_set_mean = 1290496.111745165
        bitrate_set_std = 866317.7265557338
        QoEs_mean = 4.541458676814562
        QoEs_std = 3.7425540700334294

        self.x = torch.FloatTensor()
        self.y = torch.FloatTensor()

        for file_name in os.listdir(data_dir):
            data = np.loadtxt(data_dir+"/"+file_name)

            length = data[:, 0]
            bitrate = data[:, 1]
            qoe = data[:, 2]
            buffers = data[:, 3:8]
            throughputs = data[:, 8:13]
            ssim = data[:, 13:18]
            bitrate_set = data[:, 18:23]
            QoEs = data[:, 23]

            length=length.reshape(length.shape[0],1)
            bitrate=bitrate.reshape(bitrate.shape[0],1)
            qoe=qoe.reshape(qoe.shape[0],1)
            QoEs=QoEs.reshape(QoEs.shape[0],1)

            length = (length - length_mean) / length_std
            bitrate=(bitrate-bitrate_mean)/bitrate_std
            qoe=(qoe-qoe_mean)/qoe_std
            buffers=(buffers-buffers_mean)/buffers_std
            throughputs=(throughputs-throughputs_mean)/throughputs_std
            ssim=(ssim-ssim_mean)/ssim_std
            bitrate_set=((bitrate_set-bitrate_set_mean)/bitrate_set_std)
            QoEs=((QoEs-QoEs_mean)/QoEs_std)

            data= np.concatenate((length,bitrate,qoe,buffers,throughputs,ssim,bitrate_set,QoEs),axis=1)

            self.h = data.shape[0]
            self.w = data.shape[1]

            self.x = torch.cat((self.x, torch.Tensor(data[0:self.h, 0:self.w - 1])),0)
            self.y = torch.cat((self.y, torch.Tensor(data[0:self.h, self.w - 1:self.w])),0)


    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return self.x.size(0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
    if classname.find("Linear") != -1:
        Init.xavier_uniform_(m.weight.data)

#训练数据格式：throughputs + segment_sizes + lastQoE + bufferSize + bitrate , QoEs

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden1_1 = torch.nn.Linear(1, 128) # length
        self.hidden1_2 = torch.nn.Linear(1, 128) # bitrate
        self.hidden1_3 = torch.nn.Linear(1, 128) # qoe

        self.conv4 = torch.nn.Conv1d(1, 128, 3, stride=1, padding = 1 ) # buffers
        self.conv5 = torch.nn.Conv1d(1, 128, 3, stride=1, padding = 1 ) # throughputs
        self.conv6 = torch.nn.Conv1d(1, 128, 3, stride=1, padding = 1 ) # ssim
        self.conv7 = torch.nn.Conv1d(1, 128, 3, stride=1, padding = 1 ) # bitrate_set


        self.hidden2 = torch.nn.Linear(4*128*5 + 128*3, 256)
        self.hidden3 = torch.nn.Linear(256, 128)
        self.hidden4 = torch.nn.Linear(128, 128)

        self.dropout1 = torch.nn.Dropout(p=0.5)
        self.dropout2 = torch.nn.Dropout(p=0.5)
        self.dropout3 = torch.nn.Dropout(p=0.5)

        self.predict = torch.nn.Linear(128, 1)


    def forward(self, x):
        x1 = F.relu(self.hidden1_1(x[:, 0].view(x.size(0), 1 , 1)))  # length
        x2 = F.relu(self.hidden1_2(x[:, 1].view(x.size(0), 1 , 1)))  # bitrate
        x3 = F.relu(self.hidden1_3(x[:, 2].view(x.size(0), 1 , 1)))  # qoe
        x4 = F.relu(self.conv4(x[:, 3:8].view(x.size(0), 1, -1)))       # buffers
        x5 = F.relu(self.conv5(x[:, 8:13].view(x.size(0), 1, -1)))      # throughputs
        x6 = F.relu(self.conv6(x[:, 13:18].view(x.size(0), 1, -1)))      # ssim
        x7 = F.relu(self.conv7(x[:, 18:23].view(x.size(0), 1, -1)))      # bitrate_set

        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x3 = x3.view(x3.size(0), -1)
        x4 = x4.view(x4.size(0), -1)
        x5 = x5.view(x5.size(0), -1)
        x6 = x6.view(x5.size(0), -1)
        x7 = x7.view(x5.size(0), -1)

        x8 = torch.cat((x1,x2,x3,x4,x5,x6,x7), 1)

        x = F.relu(self.dropout1(self.hidden2(x8)))
        x = F.relu(self.dropout2(self.hidden3(x)))
        x = F.relu(self.dropout3(self.hidden4(x)))

        # x = F.relu(self.hidden2(x8))
        # x = F.relu(self.hidden3(x))
        # x = F.relu(self.hidden4(x))
        #
        x = self.predict(x)
        return x


if __name__ == '__main__':

    trainingDataDir = "../sample/train"
    testDataDir = "../sample/test"

    time_now = int(time.time())
    time_local = time.localtime(time_now)
    dt = time.strftime("%Y-%m-%d_%H-%M-%S", time_local)

    modelFileDir = "../model/"+dt

    if os.path.exists(modelFileDir) == False:
        os.makedirs(modelFileDir)

    # different nets
    net_Adam = Net()
    if CUDA:
        net_Adam.cuda()
    net_Adam.apply(weights_init)
    # net_Adam.load_state_dict(torch.load('./model.pkl'))

    opt_Adam = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
    loss_func = torch.nn.MSELoss()
    if CUDA:
        loss_func.cuda()

    loss_avg_train_list = []
    loss_avg_test_list = []
    loss_avg_min_train = 1000000
    loss_avg_min_test = 1000000
    lossFileName = modelFileDir + "/" + "loss.txt"
    model_dir = modelFileDir+"/"+"models"
    if os.path.exists(model_dir) == False:
        os.makedirs(model_dir)
    figureNum = 1
    train_dataset = MyDataset(trainingDataDir)
    test_dataset = MyDataset(testDataDir)

    for epoch in range(EPOCH):

        f_loss = open(lossFileName, 'a')
        print("epoch ",epoch)

        #------------------------------------------------
        if epoch%20 == 0:
            for param_group in opt_Adam.param_groups:
                param_group['lr'] = param_group['lr'] * decay_rate

        # training------------------------------------------------
        loader = Data.DataLoader(
            dataset=train_dataset,  # torch TensorDataset format
            batch_size=BATCH_SIZE,  # mini batch size
            shuffle=True,
            num_workers=2,
        )
        loss_train_list = []
        pre_qoe_train_list = []
        real_qoe_train_list = []
        for step, (b_x_cpu, b_y_cpu) in enumerate(loader):  # for each training step
            # train your data...
            if CUDA:
                b_x = b_x_cpu.cuda()
                b_y = b_y_cpu.cuda()
            else:
                b_x = b_x_cpu
                b_y = b_y_cpu
            b_x = b_x.view(-1, inputElementCount)
            b_y = b_y.view(-1, 1)

            output = net_Adam(b_x)
            loss = loss_func(output, b_y)

            opt_Adam.zero_grad()
            loss.backward()
            opt_Adam.step()

            loss_train_list.extend(torch.abs(output.cpu().data - b_y_cpu.data).numpy().flatten())  # loss recoder
            if epoch % 20 == 0:
                pre_qoe_train_list.extend(output.cpu().data.numpy().flatten().tolist())
                real_qoe_train_list.extend(b_y_cpu.cpu().data.numpy().flatten().tolist())

        loss_avg_train = np.mean(np.array(loss_train_list))
        loss_avg_train_list.append(loss_avg_train)

        print("train: Epoch"+str(epoch)+" loss: "+str(loss_avg_train))

        if epoch % 20 == 0:
            pre_qoe_train_array=np.array(pre_qoe_train_list).reshape(len(pre_qoe_train_list),1)
            real_qoe_train_array=np.array(real_qoe_train_list).reshape(len(real_qoe_train_list),1)
            qoe_compare_array=np.concatenate((pre_qoe_train_array,real_qoe_train_array), axis=1)

            path = modelFileDir + "/qoe_compare"
            if os.path.exists(path) == False:
                os.makedirs(path)
            np.savetxt(path+"/train_qoe_"+str(epoch)+".txt", qoe_compare_array)

        #test----------------------------------------------------
        if CUDA:
            x = test_dataset.x.cuda()
            y = test_dataset.y.cuda()
        else:
            x = test_dataset.x
            y = test_dataset.y

        x = x.view(-1, inputElementCount)
        y = y.view(-1, 1)

        net_Adam.eval()
        output = net_Adam(x)
        net_Adam.train()
        loss = loss_func(output, y)

        if epoch % 200 == 0:
            pre_qoe_test_array = output.cpu().data.numpy()
            real_qoe_test_array= y.cpu().data.numpy()
            qoe_compare_array = np.concatenate((pre_qoe_test_array,real_qoe_test_array), axis=1)
            path = modelFileDir + "/qoe_compare"
            if os.path.exists(path) == False:
                os.makedirs(path)
            np.savetxt(path+"/test_qoe_"+str(epoch)+".txt", qoe_compare_array)

        loss_test_array=torch.abs(output.cpu().data - y.cpu().data).numpy()
        loss_avg_test = np.mean(loss_test_array)
        loss_avg_test_list.append(loss_avg_test)

        print("test: Epoch"+str(epoch)+" loss:"+str(loss_avg_test))
        #test end-----------------------------------------------------

        f_loss.write(str(epoch)+" "+str(loss_avg_train)+" "+str(loss_avg_test)+"\n")

        #save model
        if loss_avg_min_test > loss_avg_test or loss_avg_min_train > loss_avg_train:
            if loss_avg_min_test > loss_avg_test:
                loss_avg_min_test = loss_avg_test
            if loss_avg_min_train > loss_avg_train:
                loss_avg_min_train = loss_avg_train
            torch.save(net_Adam.state_dict(), model_dir + "/" + str(epoch) + ".pkl")




