#from __future__ import with_statement
# -*- coding: utf-8 -*
import urllib
import urllib2
import math
import os
import time
import sys #用于接收参数
import socket #用于获取IP地址
import random

TIMEOUT_COUNT = 1
IF_NO_CACHE_HEADER = False
K = 5
TURN_COUNT = 1
rtt = -1
VIDEO_RANDOM = 0
start_clock = 0
run_time = 0
#-------------------------------------
p = [-2.1484344131667874, -1.6094379124341003, -1.0986122886681098, -0.40546510810816444, 0.0] #log(R/Rmax)
ssims = [[0.9738054275512695, 0.9835991263389587, 0.9885340929031372, 0.9929453730583191, 0.9949484467506409],
        [0.9900619387626648, 0.9940679669380188, 0.9961671233177185, 0.9980104565620422, 0.9989200830459595],
        [0.8481869101524353, 0.9067575335502625, 0.9440717697143555, 0.9729890823364258, 0.9826958775520325],
        [0.9100275039672852, 0.9230990409851074, 0.9350598454475403, 0.9525502324104309, 0.9633835554122925],
        [0.9340730905532837, 0.9563283324241638, 0.9694747924804688, 0.9818214774131775, 0.9874436259269714]]
video_type = -1
#-------------------------------------
segementDuration = -1 # ms
bitrateSet = []
bitrateIndex = -1
last_bitrate_index = -1
segmentCount = -1
videoDuration = -1 # s
segmentNum = -1 #current segment index
bufferSize = -1
throughputList = []# RB
dict = {}
dict_key_list = [   'throughputList_k',
                    'downloadTime_k',
                    'chunkSize_k',
                    'ifHit_k',
                    'buffer_k',
                    'lastQoE',
                    'bitrate',
                    'rtt',
                    'chainLength',
                    'video_type',
                    'bitrate_set']

startTime = -1
csvFile = -1
totalRebufferTime = -1
START_BUFFER_SIZE = 8000 # When buffer is larger than 8s, video start to play.
MAX_BUFFER_SIZE = 30000
MIN_BUFFER_SIZE = 4000
videoName = ""

URLPrefix = "http://127.0.0.1/video"
host = "127.0.0.1"


def savefile(filepath, filename, data):
    if os.path.exists(filepath) == False:
        os.makedirs(filepath)

    file = open(filepath + "/" + filename,'w')
    file.write(data)
    file.close()


def parseMPDFile(url):
    global segementDuration
    global bitrateSet
    global bitrateIndex
    global segmentCount
    global videoDuration

    current_time = int(time.time())
    if current_time - start_clock > run_time:
        return False

    bitrateSet = []
    lineCount = 1
    VideoStartLineCount = -1
    AudioStartLineCount = -1
    segmentCount = -1
    videoDuration = -1

    request = urllib2.Request(url)
    if IF_NO_CACHE_HEADER == True:
        request.add_header('Cache-Control','no-cache')
    request.add_header('Connection', 'keep-alive')
    ifConnectionTimeout = True
    timeoutCounter = 0
    while ifConnectionTimeout == True:
        ifConnectionTimeout = False
        try:
            response = urllib2.urlopen(request)
            responseStr = response.read()
        except urllib2.URLError, e:
            # print type(e)  # not catch
            ifConnectionTimeout = True
            timeoutCounter += 1
            if timeoutCounter > TIMEOUT_COUNT:
                break
            time.sleep(0.1)
    if ifConnectionTimeout == True:
        return False

    mpdFileDir = "./video/" + videoName
    mpdFileName = "stream.mpd"
    # savefile(mpdFileDir, mpdFileName, responseStr)

    lines = responseStr.split('\n')

    for line in lines:
        if line.find("MPD mediaPresentationDuration")!=-1:
            mediaPresentationDuration = line.split('"')[1]
            mediaPresentationDuration = mediaPresentationDuration[2:len(mediaPresentationDuration)]
            if mediaPresentationDuration.find("H") != -1 :
                mediaPresentationDuration_hour = int(mediaPresentationDuration.split("H")[0])
                mediaPresentationDuration_minute = int(mediaPresentationDuration.split("H")[1].split("M")[0])
                mediaPresentationDuration_second = float(mediaPresentationDuration.split("H")[1].split("M")[1].split("S")[0])
                videoDuration = mediaPresentationDuration_hour * 3600 + mediaPresentationDuration_minute * 60 + mediaPresentationDuration_second
            elif mediaPresentationDuration.find("M")!= -1:
                mediaPresentationDuration_minute = int(mediaPresentationDuration.split("M")[0])
                mediaPresentationDuration_second = float(mediaPresentationDuration.split("M")[1].split("S")[0])
                videoDuration = mediaPresentationDuration_minute * 60 + mediaPresentationDuration_second

            else:
                mediaPresentationDuration_second = float(mediaPresentationDuration.split("S")[0])
                videoDuration = mediaPresentationDuration_second

        if line.find("Video")!=-1:
            VideoStartLineCount = lineCount
        if line.find("Audio")!=-1:
            AudioStartLineCount = lineCount
        if line.find('<SegmentTemplate')!=-1 and AudioStartLineCount == -1:
            elements = line.split(' ')
            for element in elements:
                if element.startswith("duration"):
                    segementDuration = int(element.split('"')[1])
        if line.find('<Representation')!=-1 and AudioStartLineCount == -1:
            elements = line.split(' ')
            for element in elements:
                if element.startswith("bandwidth"):
                    bitrateSet.append(int(element.split('"')[1]))
    #bitrateIndex = bitrateSet.index(min(bitrateSet))
    segmentCount =math.ceil(videoDuration / segementDuration * 1000)
    # print 'segement duration: %f' %segementDuration
    # print 'bitrateSet: ' +str(bitrateSet)
    # print 'initial bitrateIndex: %d' %bitrateIndex
    # print 'segmentCount: %d' %segmentCount
    return True


def getURL(videoName,bitrateIndex,segmentNum):
    url = URLPrefix + "/" + videoName + "/video/avc1/" + str(bitrateIndex+1)+"/seg-"+str(segmentNum)+".m4s"
    print('URL: %s' %url)
    return url


def getBitrateIndex(throughput): #RB
    global throughputList

    if len(throughputList) < 5:
        throughputList.append(throughput)
    else:
        throughputList.append(throughput)
        throughputList.pop(0)

    reciprocal = 0
    for i in range(len(throughputList)):
        reciprocal += 1/throughputList[i]
    reciprocal /= len(throughputList)

    if reciprocal!=0:
        throughputHarmonic = 1/reciprocal
    else:
        throughputHarmonic = 0

    # print("throughput harmonic: %f" % throughputHarmonic)

    for i in range(len(bitrateSet)):
        if throughputHarmonic < bitrateSet[i]:
            if i-1 < 0:
                return i
            else:
                return i-1

    return len(bitrateSet)-1

def dict2str(dict):
    s = ""

    for key in dict_key_list:
        for item in dict[key]:
            s+=str(item)+" "
    s.strip()
    return s


def startup():
    global segmentNum
    global bufferSize
    global bitrateIndex
    global last_bitrate_index
    global bitrateList
    global videoName
    global csvFile
    global dict
    global startTime

    first=True

    while bufferSize < START_BUFFER_SIZE:
        current_time = int(time.time())
        if current_time - start_clock > run_time:
            return False

        ifHit = -1
        startDownloadTime = time.time()
        url = getURL(videoName,bitrateIndex,segmentNum)
        request = urllib2.Request(url)
        #header------------------------------------------
        request.add_header('Connection', 'keep-alive')
        request.add_header('X-Debug', 'X-Cache')
        prefetch_header = dict2str(dict)
        if len(prefetch_header) != 0:
            request.add_header('Prefetch', prefetch_header)
        if IF_NO_CACHE_HEADER == True:
            request.add_header('Cache-Control', 'no-cache')
        request.add_header('bitrate', str(bitrateSet[bitrateIndex]/1024))
        #header------------------------------------------
        ifConnectionTimeout = False
        try:
            response = urllib2.urlopen(request)
            contentLength = float(response.headers['content-length'])  # 单位：B
            # print 'content-length: %f' % contentLength
            responseStr = response.read()
        except urllib2.URLError, e:
            # print type(e)  # not catch
            ifConnectionTimeout = True
        if ifConnectionTimeout == True:
            continue
        endDownloadTime = time.time()

        if ("X-Cache" in response.headers and response.headers["X-Cache"].find("miss") != -1) or IF_NO_CACHE_HEADER == True:
            ifHit = 0
        else:
            ifHit = 1

        # chunkFileDir = "./video/" + videoName + "/video/avc1/" + str(bitrateIndex+1)
        # chunkFileName = "seg-"+str(segmentNum)+".m4s"
        #savefile(chunkFileDir, chunkFileName, responseStr)

        bufferSize += segementDuration #ms
        downloadTime = endDownloadTime - startDownloadTime #s
        startTime += downloadTime
        throughput = contentLength*8/downloadTime # bps

        if last_bitrate_index == -1:
            qualityVariation = abs(0 - ssims[video_type-1][bitrateIndex])
        else:
            qualityVariation = abs(ssims[video_type-1][last_bitrate_index] - ssims[video_type-1][bitrateIndex])

        qoe = ssims[video_type-1][bitrateIndex]-qualityVariation - downloadTime
        if len(dict['chainLength'])==0:
            dict['chainLength'] = [1]
        timestamp = int(time.time())

        ss = "%d\t%d\t%d\t%d\t%d\t%f\t%f\t%f\t%d\t%d\t%d\t%d\t%d\t%f\t%d" % (
            segmentNum, contentLength, ifHit, bufferSize, bitrateSet[bitrateIndex], throughput, downloadTime,
            downloadTime, video_type, -1, -1, -1, dict['chainLength'][-1], qoe, timestamp)
        csvFile.write(ss + '\n')
        csvFile.flush()
        print("segNum\tchunkS\tHit\tbuffer\tbitrate\tthroughput\tdownloadT\trebufferT\tvtype\trate\thitty\tnet\tchainL\tqoe\ttime")
        print(ss)

        last_bitrate_index = bitrateIndex
        bitrateIndex = getBitrateIndex(throughput)
        # ----------------------
        dict['bitrate_set'] = bitrateSet
        if last_bitrate_index == bitrateIndex:
            dict['chainLength'][0] += 1
        else:
            dict['chainLength']=[1]
        if len(dict['downloadTime_k']) < K:
            dict['downloadTime_k'] = [downloadTime] * K
            dict['chunkSize_k'] = [contentLength] * K
            dict['throughputList_k'] = [throughput] * K
            dict['ifHit_k'] = [ifHit] * K
            dict['buffer_k'] = [bufferSize] * K
        else:
            dict['downloadTime_k'] = dict['downloadTime_k'][1: K] + [downloadTime]
            dict['chunkSize_k'] = dict['chunkSize_k'][1: K] + [contentLength]
            dict['throughputList_k'] = dict['throughputList_k'][1: K] + [throughput]
            dict['ifHit_k'] = dict['ifHit_k'][1: K] + [ifHit]
            dict['buffer_k'] = dict['buffer_k'][1: K] + [bufferSize]

        dict['lastQoE'] = [qoe]
        dict['bitrate'] = [str(bitrateSet[bitrateIndex])]
        dict['rtt'] =[rtt]
        dict['video_type'] = [video_type]
        # ----------------------
        segmentNum = segmentNum + 1
        if segmentNum > segmentCount:
            return True


def download():
    global segmentNum
    global bufferSize
    global bitrateIndex
    global last_bitrate_index
    global videoName
    global csvFile
    global totalRebufferTime
    global dict

    while True:
        current_time = int(time.time())
        if current_time - start_clock > run_time:
            return

        if bufferSize + segementDuration <= MAX_BUFFER_SIZE:  # download new segment
            ifHit = -1
            startDownloadTime = time.time() # s
            url = getURL(videoName, bitrateIndex, segmentNum)
            request = urllib2.Request(url)
            # header------------------------------------------
            request.add_header('Connection', 'keep-alive')
            request.add_header('X-Debug', 'X-Cache')
            prefetch_header = dict2str(dict)
            request.add_header('Prefetch', prefetch_header)
            if IF_NO_CACHE_HEADER == True:
                request.add_header('Cache-Control', 'no-cache')
            request.add_header('bitrate', str(bitrateSet[bitrateIndex] / 1024))
            # header------------------------------------------
            ifConnectionTimeout = False
            try:
                response = urllib2.urlopen(request)
                contentLength = float(response.headers['content-length'])  # B
                # print 'content-length: %f' % contentLength
                responseStr = response.read()
            except urllib2.URLError, e:
                # print type(e)  # not catch
                ifConnectionTimeout = True
            if ifConnectionTimeout == True:
                continue
            endDownloadTime = time.time()

            if "X-Cache" in response.headers:
                # print(response.headers)
                if response.headers["X-Cache"].find("miss") != -1 or IF_NO_CACHE_HEADER == True:
                    ifHit = 0
                else:
                    ifHit = 1
            downloadTime = endDownloadTime - startDownloadTime # s
            #--------------------------------------------------
            rebufferTimeOneSeg = -1
            if downloadTime*1000 > bufferSize:
                rebufferTimeOneSeg = downloadTime - bufferSize/1000 # s
                bufferSize = 0
                totalRebufferTime += rebufferTimeOneSeg
            else:
                bufferSize = bufferSize - downloadTime*1000 # ms
                rebufferTimeOneSeg = 0
            bufferSize += segementDuration
            #--------------------------------------------------
            throughput = contentLength*8 / downloadTime  # bps
            qualityVariation = abs(ssims[video_type-1][last_bitrate_index] - ssims[video_type-1][bitrateIndex])
            qoe = ssims[video_type-1][bitrateIndex] - qualityVariation - rebufferTimeOneSeg
            #---------------------------------------------------
            timestamp = int(time.time())
            ss = "%d\t%d\t%d\t%d\t%d\t%f\t%f\t%f\t%d\t%d\t%d\t%d\t%d\t%f\t%d" % (
                segmentNum, contentLength, ifHit, bufferSize, bitrateSet[bitrateIndex], throughput, downloadTime,
                rebufferTimeOneSeg, video_type, -1, -1, -1, dict['chainLength'][-1], qoe, timestamp)
            csvFile.write(ss + '\n')
            csvFile.flush()
            print("segNum\tchunkS\tHit\tbuffer\tbitrate\tthroughput\tdownloadT\trebufferT\tvtype\trate\thitty\tnet\tchainL\tqoe\ttime")
            print(ss)
            #---------------------------------------------------
            last_bitrate_index = bitrateIndex
            if bufferSize < MIN_BUFFER_SIZE:
                bitrateIndex = 0
            else:
                bitrateIndex = getBitrateIndex(throughput)

            # ----------------------------------------
            dict['bitrate_set'] = bitrateSet
            if last_bitrate_index == bitrateIndex:
                dict['chainLength'][0] += 1
            else:
                dict['chainLength']=[1]
            if len(dict['downloadTime_k']) < K:
                dict['downloadTime_k'] = [downloadTime] * K
                dict['chunkSize_k'] = [contentLength] * K
                dict['throughputList_k'] = [throughput] * K
                dict['ifHit_k'] = [ifHit] * K
                dict['buffer_k'] = [bufferSize] * K
            else:
                dict['downloadTime_k'] = dict['downloadTime_k'][1: K] + [downloadTime]
                dict['chunkSize_k'] = dict['chunkSize_k'][1: K] + [contentLength]
                dict['throughputList_k'] = dict['throughputList_k'][1: K] + [throughput]
                dict['ifHit_k'] = dict['ifHit_k'][1: K] + [ifHit]
                dict['buffer_k'] = dict['buffer_k'][1: K] + [bufferSize]

            dict['lastQoE'] = [qoe]
            dict['bitrate'] = [str(bitrateSet[bitrateIndex])]
            dict['rtt'] = [rtt]
            dict['video_type'] = [video_type]
            # ----------------------------------------

            segmentNum = segmentNum + 1
            if segmentNum > segmentCount:
                break
        else:
            bufferSize = MAX_BUFFER_SIZE - segementDuration
    return True


def main():
    global segementDuration
    global bitrateSet
    global bitrateIndex
    global last_bitrate_index
    global segmentCount
    global videoDuration
    global segmentNum
    global bufferSize
    global throughputList
    global startTime
    global csvFile
    global totalRebufferTime
    global videoName
    global TURN_COUNT
    global IF_NO_CACHE_HEADER
    global class_id
    global video_type
    global rtt
    global dict
    global start_clock
    global run_time

    start_clock = int(time.time())
    traceIndex = 0
    client_id = 1
    TURN_COUNT = 1
    run_time = 1000
    VIDEO_RANDOM=0
    for i in range(1, len(sys.argv)):
        if i == 1:
            traceIndex = int(sys.argv[i])
        if i == 2:
            TURN_COUNT = int(sys.argv[i])
        if i == 3:
            run_time = int(sys.argv[i]) #s
        if i == 4:
            VIDEO_RANDOM = int(sys.argv[i])

    csvFileDir = "./client/trace/trace"+str(traceIndex)

    #ping to get rtt--------------------------
    # p= ping.ICMPPing()
    # rtt = p.main(host)
    #--------------------------------------------------------------
    video_popularity_file = open("./video.txt")
    video_popularity_list = video_popularity_file.readlines()
    video_popularity_list = [(i.split(" ")[0],float(i.split(" ")[1]),int(float(i.split(" ")[2]))) for i in video_popularity_list] #(video_name, popularity, video_tupe)
    #---------------------------------------------------------------
    for turn in range(TURN_COUNT):
        for i in range(1):
            if os.path.exists(csvFileDir) == False:
                os.makedirs(csvFileDir)

            segementDuration = -1  # unit:ms
            bitrateSet = []
            bitrateIndex = 0
            last_bitrate_index = -1
            segmentCount = -1
            videoDuration = -1  # unit:s
            segmentNum = 1  # current segment index
            bufferSize = 0
            throughputList = []  # RB
            startTime = -1
            csvFile = -1
            totalRebufferTime = 0
            videoName = ""
            video_type = 0
            dict = {'throughputList_k': [],
                    'downloadTime_k': [],
                    'chunkSize_k': [],
                    'ifHit_k': [],
                    'buffer_k':[],
                    'lastQoE': [],
                    'bitrate': [],
                    'rtt': [],
                    'chainLength': [],
                    'video_type':[],
                    'bitrate_set':[]}
            # ---------------------------------------------------------------
            if VIDEO_RANDOM:
                video_random = random.random()
                videoName = ""
                for i in range(len(video_popularity_list)):
                    if video_random < video_popularity_list[i][1]:
                        videoName = video_popularity_list[i-1][0]
                        video_type = video_popularity_list[i-1][2]
                        break
                if videoName == "":
                    videoName = video_popularity_list[-1][0]
                    video_type = 1
            else:
                videoName = "yourname"
                video_type = 1
            mpd_url = URLPrefix+"/"+videoName+"/stream.mpd"
            ifSuccess = parseMPDFile(mpd_url)
            if ifSuccess == False:
                return
            # ---------------------------------------------------------------
            time_now = int(time.time())
            time_local = time.localtime(time_now)
            dt = time.strftime("%Y-%m-%d_%H-%M-%S", time_local)
            ran = random.random()
            if time_now - start_clock > run_time:
                return

            # myname = socket.getfqdn(socket.gethostname())
            # myaddr = socket.gethostbyname(myname)
            # print("myaddr=%s", myaddr)
            # print("myname=%s", myname)

            # csvFileName = csvFileDir +"/"+ str(dt) +"_" + videoName +"_"+ str(myaddr)+ ".csv"
            csvFileName = csvFileDir +"/"+ str(dt) +"_" + videoName +".csv"
            csvFile = open(csvFileName, 'w')
            csvFile.write("segNum\tsize\tHit\tbuffer\tbitrate\tthroughput\tdownloadT\trebufferT\t"
                  "vtype\tratelimit\thittype\tnetwork\tchainLen\tqoe\ttime\n")

            startTime = 0
            ifSuccess = startup()
            if ifSuccess == False:
                return
            startupLatency = startTime
            download()

            csvFile.close()
            print("play complete")
            print("startupLatency=" + str(startupLatency))
            print("total rebuffering time=" + str(totalRebufferTime) +"ms")



main()