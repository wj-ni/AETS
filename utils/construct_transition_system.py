import os
import time
import datetime
import numpy as np
from datetime import datetime

# abstraction1 : sequence
def abstraction_sequence(state_dict, edge_list, edgeCount_list, traceId_list, eventId_list):
    for i in range(len(traceId_list) - 1):
        prefix = eventId_list[:i + 1]
        next = eventId_list[:i + 2]
        prefixIndex = -1
        nextIndex = -1
        for key in state_dict.keys():
            if state_dict[key] == prefix:
                prefixIndex = key
            if state_dict[key] == next:
                nextIndex = key
        if prefixIndex == -1:
            prefixIndex = len(state_dict)
            state_dict[prefixIndex] = prefix
        if nextIndex == -1:
            nextIndex = len(state_dict)
            state_dict[nextIndex] = next
        if [prefixIndex, nextIndex] not in edge_list:
            edge_list.append([prefixIndex, nextIndex])
            edgeCount_list.append([[prefixIndex, nextIndex], 1])
        else:
            for edgeCount in edgeCount_list:
                if edgeCount[0] == [prefixIndex, nextIndex]:
                    edgeCount[1] = edgeCount[1] + 1
    return state_dict, edge_list, edgeCount_list


# abstraction3 : multi-set
def abstraction_multiSet(state_dict, edge_list, edgeCount_list, traceId_list, eventId_list):
    # print("====================")
    # print(traceId_list)
    # print(eventId_list)

    for i in range(len(traceId_list) - 1):
        prefix = {}
        next = {}
        for node in eventId_list[:i + 1]:
            # print("===============")
            # print(node)
            if node in prefix.keys():
                prefix[node] = prefix[node] + 1
            else:
                prefix[node] = 1
            if node in next.keys():
                next[node] = next[node] + 1
            else:
                next[node] = 1
        if eventId_list[i + 1] in next.keys():
            next[eventId_list[i + 1]] = next[eventId_list[i + 1]] + 1
        else:
            next[eventId_list[i + 1]] = 1
        prefixIndex = -1
        nextIndex = -1
        # print(state_dict)
        for key in state_dict.keys():
            if state_dict[key] == prefix:
                prefixIndex = key
        if prefixIndex == -1:
            prefixIndex = len(state_dict)
            state_dict[prefixIndex] = prefix
        for key in state_dict.keys():
            if state_dict[key] == next:
                nextIndex = key
        if nextIndex == -1:
            nextIndex = len(state_dict)
            state_dict[nextIndex] = next
        # print(state_dict)
        if [prefixIndex, nextIndex] not in edge_list:
            edge_list.append([prefixIndex, nextIndex])
            edgeCount_list.append([[prefixIndex, nextIndex], 1])
        else:
            for edgeCount in edgeCount_list:
                if edgeCount[0] == [prefixIndex, nextIndex]:
                    edgeCount[1] = edgeCount[1] + 1
    return state_dict, edge_list, edgeCount_list


# abstraction3 : set
def abstraction_set(state_dict, edge_list, edgeCount_list, traceId_list, eventId_list):
    for i in range(len(traceId_list) - 1):
        prefix = set(eventId_list[:i + 1])
        next = set(eventId_list[:i + 2])
        prefixIndex = -1
        nextIndex = -1
        for key in state_dict.keys():
            if state_dict[key] == prefix:
                prefixIndex = key
        if prefixIndex == -1:
            prefixIndex = len(state_dict)
            state_dict[prefixIndex] = prefix
        for key in state_dict.keys():
            if state_dict[key] == next:
                nextIndex = key
        if nextIndex == -1:
            nextIndex = len(state_dict)
            state_dict[nextIndex] = next
        if [prefixIndex, nextIndex] not in edge_list:
            edge_list.append([prefixIndex, nextIndex])
            edgeCount_list.append([[prefixIndex, nextIndex], 1])
        else:
            for edgeCount in edgeCount_list:
                if edgeCount[0] == [prefixIndex, nextIndex]:
                    edgeCount[1] = edgeCount[1] + 1
    return state_dict, edge_list, edgeCount_list


def readFile(file, func,time_unit):
    fp = open(file, "r", encoding='utf-8')
    next(fp)
    train_list = []
    edge_list = []
    state_dict = {}
    edgeCount_list = []
    trace_log = fp.readlines()
    num_train = 0
    trace_temp = []
    event_list = []
    max = 0
    current_traceId = trace_log[0].split(",")[0]
    for line in trace_log:
        traceId = line.split(",")[0]
        if current_traceId == traceId:
            trace_temp.append(line)
        else:
            current_traceId = traceId
            num_train += 1
            if len(trace_temp) > max:
                max = len(trace_temp)
            train_list.append(trace_temp)
            trace_temp = []
            trace_temp.append(line)

    train_list.append(trace_temp)
    print("=============max==============")
    print(max)
    # for index in range(0,len(traceTransList)):
    #     # print(traceTransList[index])
    #     # print("=======================")
    #     for length in range(1,len(traceTransList[index])+1):
    #         tracePrefix = traceTransList[index][:length]
    #         timePrefix = timeTransList[index][length-1]
    #         # print(tracePrefix,timePrefix)
    #         if tracePrefix in traceDifferLengthDict[length]:
    #             tracePrefixIndex = traceDifferLengthDict[length].index(tracePrefix)
    #             timeDifferLengthDict[length][tracePrefixIndex].append(timePrefix)
    #         else:
    #             traceDifferLengthDict[length].append(tracePrefix)
    #             timeDifferLengthDict[length].append([timePrefix])
    # 遍历流程日志 训练集
    for trace in train_list:
        traceId_list = []
        eventId_list = []
        for event in trace:
            traceId, eventId, time = event.split(",")
            if eventId not in event_list:
                event_list.append(eventId)
            traceId_list.append(traceId)
            eventId_list.append(eventId)
            # time_list.append(time)
        # sequence
        if func == 'sequence':
            state_dict, edge_list, edgeCount_list = abstraction_sequence(state_dict, edge_list, edgeCount_list,
                                                                         traceId_list, eventId_list)
        # set
        elif func == 'set':
            state_dict, edge_list, edgeCount_list = abstraction_set(state_dict, edge_list, edgeCount_list, traceId_list,
                                                                    eventId_list)
        #multiSet
        elif func == 'multiSet':
            state_dict, edge_list, edgeCount_list = abstraction_multiSet(state_dict, edge_list, edgeCount_list, traceId_list,
                                                                    eventId_list)
    return state_dict, edge_list, edgeCount_list,event_list


# rmse计算
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


# 获取指定文件夹下面所有文件名
def file_name(path):
    F = []
    for root, dirs, files in os.walk(path):
        # print root
        # print dirs
        for file in files:
            # print file.decode('gbk')    #文件名中有中文字符时转码
            if os.path.splitext(file)[1] == '.txt':
                t = os.path.splitext(file)[0]
                F.append(t)  # 将所有的文件名添加到L列表中
    return F  # 返回L列表


def save_data(state_dict, edge_list, edgeCount_dict):
    path = "./save/"
    with open(path + "index2state_BPI2012.txt", "a", encoding='utf-8') as f:
        for key in state_dict.keys():
            f.write(str(key) + "\t" + str(state_dict[key]) + "\n")
    with open(path + "edge_BPI2012.edgelist", "a", encoding="utf-8") as rf:
        edge_list = sorted(edge_list, key=lambda x: x[0])
        # print(edge_list)
        for edge in edge_list:
            rf.writelines(str(edge[0]) + "\t" + str(edge[1]) + "\n")
def save_data_weight(state_dict, edge_list, edgeCount_dict):
    path = "./save/"
    with open(path + "index2state_BPI2012.txt", "a", encoding='utf-8') as f:
        for key in state_dict.keys():
            f.write(str(key) + "\t" + str(state_dict[key]) + "\n")
    with open(path + "edge_BPI2012.edgelist", "a", encoding="utf-8") as rf:
        edge_list = sorted(edge_list, key=lambda x: x[0])
        for key in state_dict.keys():
            relate_list = []
            allNumber = 0
            for edge in edgeCount_dict:
                if edge[0][0] == key:
                    relate_list.append(edge)
                    allNumber = allNumber + edge[1]
            for edge in relate_list:
                rf.writelines(str(edge[0][0]) + "\t" + str(edge[0][1])+ "\t" + str(format(float(edge[1])/float(allNumber),'.4f')) + "\n")



if __name__ == '__main__':
    # 流程日志
    path = "./newdata/"
    file = path + "helpdesk.csv"
    # state_dict, edge_list, edgeCount_dict = readFile(file, 'sequence')
    # save_data(state_dict, edge_list, edgeCount_dict)
    # # save_data_weight(state_dict, edge_list, edgeCount_dict)
    # print(state_dict)
    # print(edge_list)
    # print("============")
    # print(edgeCount_dict)
    state_dict, edge_list, edgeCount_list, event_list = readFile("./data/ogrinal/BPI_Challenge_2012_W.csv",'multiSet')
