#segNum size Hit buffer bitrate throughput downloadT rebufferT vtype ratelimit hittype network chainLen qoe
import os

ssims = [[0.9738054275512695, 0.9835991263389587, 0.9885340929031372, 0.9929453730583191, 0.9949484467506409],
        [0.9900619387626648, 0.9940679669380188, 0.9961671233177185, 0.9980104565620422, 0.9989200830459595],
        [0.8481869101524353, 0.9067575335502625, 0.9440717697143555, 0.9729890823364258, 0.9826958775520325],
        [0.9100275039672852, 0.9230990409851074, 0.9350598454475403, 0.9525502324104309, 0.9633835554122925],
        [0.9340730905532837, 0.9563283324241638, 0.9694747924804688, 0.9818214774131775, 0.9874436259269714]]


TIME_SCALE = 5
dict = {"segNum":0, "size":1, "Hit":2, "buffer":3, "bitrate":4, "throughput":5, "downloadT":6, "rebufferT":7, "vtype":8, "ratelimit":9, "hittype":10, "network":11, "chainLen":12, "qoe":13}

trace_dir = "./trace"
fileIndex = '1'
print("fileIndex=", fileIndex)
qoe_train_sample_dir = "./sample/train"
qoe_test_sample_dir = "./sample/test"
if os.path.exists(qoe_train_sample_dir) == False:
    os.makedirs(qoe_train_sample_dir)
if os.path.exists(qoe_test_sample_dir) == False:
    os.makedirs(qoe_test_sample_dir)

train_sample_file = open(qoe_train_sample_dir + "/"+ "train_sample_"+str(fileIndex)+".dat", 'w')
test_sample_file = open(qoe_test_sample_dir + "/"+ "test_sample_"+str(fileIndex)+".dat", 'w')

train_counter = 1 # 每10个train sample，2个test sample
test_counter = 1

video_bitrate_set_dict = {}
lines = open("./bitrate_set.txt").readlines()
for line in lines:
    video_name = line.split(',')[0]
    bitrate_set = line.split(',')[1:]
    bitrate_set = [int(i.strip()) for i in bitrate_set]
    video_bitrate_set_dict[video_name] = bitrate_set

for client_dir in os.listdir(trace_dir):
    for trace_name in os.listdir(trace_dir+"/"+client_dir):
        file_trace = open(trace_dir+"/"+client_dir+"/"+trace_name)
        # video_name = trace_name.split("_")[0]
        video_name = trace_name.split("_")[2]
        print("video_name=",video_name)
        ssim = []

        bitrate_sets = str(video_bitrate_set_dict[video_name])[1:-1].replace(", "," ")

        lines = file_trace.readlines()
        outliers_flag = False
        for currentLineIndex in range(1,len(lines)-(TIME_SCALE+1+5+2)):
            length = ""
            bitrate =""
            qoe =""
            buffers = ""
            throughputs = ""
            ssim = ""
            QoEs = ""

            for lineIndex in range(currentLineIndex,currentLineIndex+TIME_SCALE):
                line = lines[lineIndex]
                element = line.split('\t')

                throughputs = throughputs + element[dict["throughput"]] + " "
                buffers = buffers + element[dict["buffer"]] + " "

                if lineIndex == currentLineIndex + TIME_SCALE -1:
                    length = element[dict["size"]] + " "
                    qoe = element[dict["qoe"]].strip('\n') + " "
                    if float(qoe) <-100:
                        outliers_flag = True
                    vtype = int(element[dict["vtype"]])
                    ssim = str(ssims[vtype-1])[1:-1].replace(", "," ") + " "

            bitrate = lines[currentLineIndex+TIME_SCALE].split('\t')[dict["bitrate"]] + " "
            inputString = length + bitrate + qoe + buffers + throughputs + ssim + bitrate_sets.strip()+" "

            outputQoE = 0
            for lineIndex in range(currentLineIndex+TIME_SCALE+1, currentLineIndex+TIME_SCALE+6):
                line = lines[lineIndex]
                element = line.split('\t')
                qoe = float(element[dict["qoe"]])
                if qoe < -100:
                    outliers_flag = True
                outputQoE += qoe

            sample = inputString + str(outputQoE)
            if outliers_flag:
                outliers_flag = False
                continue
            if train_counter <= 10:
                train_sample_file.write(sample + "\n")
                train_counter +=1
            else:
                test_sample_file.write(sample + "\n")
                test_counter += 1
                if test_counter == 3:
                    train_counter = 1
                    test_counter = 1

        file_trace.close()

train_sample_file.close()
test_sample_file.close()
