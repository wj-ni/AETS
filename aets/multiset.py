import torch
import torch.nn as nn
import copy
import torch.optim as optim
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from model.MLP import MLP
from model.autoencoder import autoencoder
from utils.construct_transition_system import *
from utils.metrics import *

dir_checkpoint = './checkpoints'


def getTrainAndTestData(file, type, state_dict, event_list):
    initTrans = []
    for i in range(0, len(event_list)):
        initTrans.append(0)
    fp = open('./data/' + file, "r", encoding='utf-8')
    next(fp)
    data_list = []
    trace_temp = []
    trace_log = fp.readlines()
    current_traceId = trace_log[0].split(",")[0]
    for line in trace_log:
        traceId = line.split(",")[0]
        if current_traceId == traceId:
            trace_temp.append(line)
        else:
            current_traceId = traceId
            data_list.append(trace_temp)
            trace_temp = []
            trace_temp.append(line)
    data_list.append(trace_temp)
    train_list, test_list = train_test_split(data_list, train_size=0.7, test_size=0.3, random_state=123)
    train_vec = []
    train_label = []
    train_event = []
    initAttr = []
    tempVec = [int(x) for x in data_list[0][0].split(",")[2:-1]]
    for i in range(0, len(tempVec)):
        initAttr.append(0)
    # train dataset construct
    for trace in train_list:
        for event in trace[:-1]:
            remainTime = float(event.split(",")[-1].replace("\n", ""))
            currVec = [int(x) for x in event.split(",")[2:-1]]
            currEventIndex = trace.index(event)
            transVec = initTrans.copy()
            if currEventIndex >= 2:
                tempEvent = []
                for i in range(0, currEventIndex + 1):
                    event_ = trace[i].split(",")[1]
                    currEventTempIndex = event_list.index(event_)
                    transVec[currEventTempIndex] += 1
                    tempEvent.append(event_)
                lastVec1 = [int(x) for x in trace[currEventIndex - 1].split(",")[2:-1]]
                lastVec2 = [int(x) for x in trace[currEventIndex - 2].split(",")[2:-1]]
                train_vec.append(currVec + lastVec1 + lastVec2 + transVec)
                train_label.append(remainTime)
                tempEvent.sort()
                train_event.append(tempEvent)
    test_vec = []
    test_label = []
    test_event = []
    # test dataset construct
    for trace in test_list:
        for event in trace[:-1]:
            remainTime = float(event.split(",")[-1].replace("\n", ""))
            currVec = [int(x) for x in event.split(",")[2:-1]]
            currEventIndex = trace.index(event)
            transVec = initTrans.copy()
            if currEventIndex >= 2:
                tempEvent = []
                for i in range(0, currEventIndex + 1):
                    event_ = trace[i].split(",")[1]
                    currEventTempIndex = event_list.index(event_)
                    transVec[currEventTempIndex] += 1
                    tempEvent.append(event_)
                lastVec1 = [int(x) for x in trace[currEventIndex - 1].split(",")[2:-1]]
                lastVec2 = [int(x) for x in trace[currEventIndex - 2].split(",")[2:-1]]
                test_vec.append(currVec + lastVec1 + lastVec2 + transVec)
                test_label.append(remainTime)
                tempEvent.sort()
                test_event.append(tempEvent)
    return train_vec, test_vec, train_label, test_label, train_event, test_event


def train(file, ts_type, time_unit):
    state_dict, edge_list, edgeCount_list, event_list = readFile("./data/orginal/" + file, ts_type, time_unit)
    stateIndex = list(state_dict.keys())
    stateValue = []
    for state in state_dict.values():
        tem1 = []
        for key in state.keys():
            for num in range(0, state[key]):
                tem1.append(key)
        tem1.sort()
        stateValue.append(tem1)
    train_vec, test_vec, train_label, test_label, train_event, test_event = getTrainAndTestData(file, ts_type,
                                                                                                state_dict, event_list)

    tensor_x = torch.from_numpy(np.array(train_vec).astype(np.float32)).cuda()
    tensor_y = torch.from_numpy(np.array(train_label).astype(np.float32)).cuda()
    my_dataset = TensorDataset(tensor_x, tensor_y)
    my_dataset_loader = DataLoader(my_dataset, batch_size=128, shuffle=False)
    model = autoencoder()
    criterion = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(300):
        total_loss = 0
        for i, (x, y) in enumerate(my_dataset_loader):
            optimizer.zero_grad()
            _, pred = model(Variable(x.cuda()).cuda())
            loss = criterion(pred, x)
            loss.backward()
            optimizer.step()
            total_loss += loss
        if epoch % 100 == 0:
            print(str(total_loss.data))
    trainLowDim = []
    testLowDim = []
    for x in train_vec:
        x_ = Variable(torch.Tensor(np.array(x)).cuda()).cuda()
        _, pred = model(Variable(x_.cuda()).cuda())
        trainLowDim.append(_.detach().cpu().numpy().tolist())
    for x in test_vec:
        x_ = Variable(torch.Tensor(np.array(x)).cuda()).cuda()
        _, pred = model(Variable(x_).cuda())
        testLowDim.append(_.detach().cpu().numpy().tolist())

    singleTrainVec = {}
    singleTrainLabel = {}
    modelDict = {}
    for index in stateIndex:
        singleTrainVec[index] = []
        singleTrainLabel[index] = []
    for eventIndex in range(0, len(train_event)):
        currStateIndex = stateValue.index(train_event[eventIndex])
        singleTrainVec[currStateIndex].append(trainLowDim[eventIndex])
        singleTrainLabel[currStateIndex].append(train_label[eventIndex])
    singleTestVec = {}
    singleTestLabel = {}
    for index in stateIndex:
        singleTestVec[index] = []
        singleTestLabel[index] = []
    for eventIndex in range(0, len(test_event)):
        currStateIndex = stateValue.index(test_event[eventIndex])
        singleTestVec[currStateIndex].append(testLowDim[eventIndex])
        singleTestLabel[currStateIndex].append(test_label[eventIndex])
    tensor_x = torch.from_numpy(np.array(trainLowDim).astype(np.float32)).cuda()
    tensor_y = torch.from_numpy(np.array(train_label).astype(np.float32)).cuda()
    my_dataset = TensorDataset(tensor_x, tensor_y)
    my_dataset_loader = DataLoader(my_dataset, batch_size=128, shuffle=False)
    modelTotal = MLP()
    criterionLinear = nn.L1Loss().cuda()
    optimizerLinear = optim.Adam(modelTotal.parameters(), lr=0.0001)
    for epoch in range(100):
        print("total train epoch:" + str(epoch))
        total_loss = 0
        for i, (x, y) in enumerate(my_dataset_loader):
            pred = modelTotal(Variable(x.cuda()).cuda())
            pred = pred.squeeze(-1)
            loss = criterionLinear(pred, y)
            optimizerLinear.zero_grad()
            loss.backward()
            optimizerLinear.step()
            total_loss += loss
    # transfer learning train
    for index in singleTrainVec.keys():
        if len(singleTrainVec[index]) < 10:
            continue
        tensor_x = torch.from_numpy(np.array(singleTrainVec[index]).astype(np.float32)).cuda()
        tensor_y = torch.from_numpy(np.array(singleTrainLabel[index]).astype(np.float32)).cuda()
        my_dataset = TensorDataset(tensor_x, tensor_y)
        my_dataset_loader = DataLoader(my_dataset, batch_size=8, shuffle=False)
        modelLinear = copy.deepcopy(modelTotal)
        criterionLinear = nn.L1Loss().cuda()
        optimizerLinear = optim.Adam(modelLinear.parameters(), lr=0.00001)
        for epoch in range(100):
            total_loss = 0
            for i, (x, y) in enumerate(my_dataset_loader):
                pred = modelLinear(Variable(x.cuda()).cuda())
                pred = pred.squeeze(-1)
                loss = criterionLinear(pred, y)
                optimizerLinear.zero_grad()
                loss.backward()
                optimizerLinear.step()
                total_loss += loss
            print("total_loss = " + str(total_loss))
        torch.save(modelLinear.state_dict(),
                   dir_checkpoint + "/model_" + file.split('.')[0] + "_" + ts_type + "_" + f'_state{index}.pth')
        modelDict[index] = modelLinear
    return singleTrainVec, singleTrainLabel, singleTestVec, singleTestLabel, modelDict


def predict(singleTrainVec, singleTrainLabel, singleTestVec, singleTestLabel, modelDict, ts_type):
    # predict
    predList = []
    realList = []
    fout = open(file.split(".")[0] + "_" + ts_type + "_state_our.csv", "a", encoding='utf-8')
    fout.write("stateId,state,trainNumber,testNumber,MSE,MAE\n")
    for index in singleTestVec.keys():
        singlepred = []
        singlereal = []
        if index in modelDict.keys():
            for vec in range(0, len(singleTestVec[index])):
                input = Variable(torch.Tensor(np.array(singleTestVec[index][vec])).cuda()).cuda()
                pred = modelDict[index](input)
                # pred = abs(pred)
                predList.append(abs(pred))
                realList.append(singleTestLabel[index][vec])
                singlepred.append(abs(pred))
                singlereal.append(singleTestLabel[index][vec])
            if len(singlepred) != 0:
                MSE = computeMSE(singlereal, singlepred)
                MAE = computeMAE(singlereal, singlepred)
                state = str(state_dict[index]).replace(",", " ")
                fout.write(str(index) + "," + state + "," + str(len(singleTrainVec[index])) + "," + str(
                    len(singleTestVec[index])) + "," + str(float(MSE)) + "," + str(float(MAE)) + "\n")
                # print("===========" + str(index) + "===============")
                # print(MSE, MAE)
    MSE = computeMSE(realList, predList)
    MAE = computeMAE(realList, predList)
    fout.write("totalMSE:," + str(MSE) + "\n")
    fout.write("totalMAE:," + str(MAE) + "\n")
    return MAE, MSE


def train_multiset(file, ts_type, time_unit):
    # train
    singleTrainVec, singleTrainLabel, singleTestVec, singleTestLabel, modelDict = train(file, ts_type, time_unit)
    # predict
    MAE, MSE = predict(singleTrainVec, singleTrainLabel, singleTestVec, singleTestLabel, modelDict, ts_type)
    return MAE, MSE
