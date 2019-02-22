import torch
import torchvision
import torch.nn.functional as F
import numpy as np


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

        # x = F.relu(self.dropout1(self.hidden2(x8)))
        # x = F.relu(self.dropout2(self.hidden3(x)))
        # x = F.relu(self.dropout3(self.hidden4(x)))

        x = F.relu(self.hidden2(x8))
        x = F.relu(self.hidden3(x))
        x = F.relu(self.hidden4(x))
        #
        x = self.predict(x)
        return x

if __name__ == '__main__':

    # An instance of your model.
    net_Adam = Net()
    net_Adam.load_state_dict(torch.load('./model.pkl'))

    # An example input you would normally provide to your model's forward() method.

    data_list = [1394719.061683, 1426161.330388, 1287996.198608, 1306128.811535, 1380105.530650, 1321164.839405,
               1356234.005458, 1286841.231577, 318137, 348205, 406702, 423753, 415547, 411728, 442243, 364343,
               972.536133, 29997, 995877]
    data_list = [1038226, 2618238, 0.998920,
                 27176, 26807, 26839, 27268, 27334,
                 3144012.280313, 3111476.463350,3127473.335923, 3170582.046790, 3115589.897671,
                 0.9900619387626648, 0.9940679669380188, 0.9961671233177185, 0.9980104565620422, 0.9989200830459595,
                 338304, 574892, 924838, 1907890, 2618238]
    data_list = [1331319,2849990,0.982696,
                 28391,26377, 24661,27525,25647,
                 9567508.054229,2336783.999028,2509846.033833,6524987.781939,2446835.756780,
                 0.8481869101524353,0.9067575335502625,0.9440717697143555, 0.9729890823364258, 0.9826958775520325,
                 358494, 597824, 1004533, 1963121, 2849990, 4.884359]

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

    data = np.array(data_list).reshape(1,-1)

    length = data[:, 0]
    bitrate = data[:, 1]
    qoe = data[:, 2]
    buffers = data[:, 3:8]
    throughputs = data[:, 8:13]
    ssim = data[:, 13:18]
    bitrate_set = data[:, 18:23]

    length = length.reshape(length.shape[0], 1)
    bitrate = bitrate.reshape(bitrate.shape[0], 1)
    qoe = qoe.reshape(qoe.shape[0], 1)

    length = (length - length_mean) / length_std
    bitrate = (bitrate - bitrate_mean) / bitrate_std
    qoe = (qoe - qoe_mean) / qoe_std
    buffers = (buffers - buffers_mean) / buffers_std
    throughputs = (throughputs - throughputs_mean) / throughputs_std
    ssim = (ssim - ssim_mean) / ssim_std
    bitrate_set = ((bitrate_set - bitrate_set_mean) / bitrate_set_std)

    data = np.concatenate((length, bitrate, qoe, buffers, throughputs, ssim, bitrate_set), axis=1)

    data_list=data.flatten().tolist()
    print(data_list)
    example = torch.tensor([data_list]) # 19维
    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    traced_script_module = torch.jit.trace(net_Adam, example)
    traced_script_module.save("./model.pt")