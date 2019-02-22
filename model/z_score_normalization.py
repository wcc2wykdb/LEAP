import numpy as np
import os

def cal_normal_parameter(data_name,data):
    mean = np.average(data)
    std = np.std(data)
    print("{}_mean={}".format(data_name,mean))
    print("{}_std={}".format(data_name,std))

sample_data = np.loadtxt("./train_sample.dat")
length = sample_data[:, 0]
bitrate = sample_data[:, 1]
qoe = sample_data[:, 2]
buffers = sample_data[:, 3:8]
throughputs = sample_data[:, 8:13]
ssim = sample_data[:,13:18]
bitrate_set = sample_data[:,18:23]
QoEs = sample_data[:,23]

cal_normal_parameter("length",length)
cal_normal_parameter("bitrate",bitrate)
cal_normal_parameter("qoe",qoe)
cal_normal_parameter("buffers",buffers)
cal_normal_parameter("throughputs",throughputs)
cal_normal_parameter("ssim",ssim)
cal_normal_parameter("bitrate_set",bitrate_set)
cal_normal_parameter("QoEs",QoEs)
