import torch
import numpy as np

from DFL4ATC import DeepATC, args
from sklearn.metrics import roc_auc_score, matthews_corrcoef
import copy
from utils import DataLoader, DataTransfer, mini_batch, matric

import sys
import torch.nn as nn
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import json
drug_data = 'train_data/drugATC.json'  # {'drugID': {'vector': '0.1,0.2,...', 'SMILES': 'CC', 'label': '[1, 0, ..]'}}
cv_json = 'train_data/CV_ATC.json'  # cross validation data index
data_loader = DataLoader(drug_data, cv_json)
drug_dict = data_loader.data_dict
CROSS_FOLD = 4
x_train, x_validation, y_train, y_validation = data_loader.split(CROSS_FOLD, CV=True, test_size=0.3)

data_transfer = DataTransfer()

print('Train:', len(y_train))
print('Test:', len(y_validation))

#  init param
best_auc = 0
avg_loss = 0
num_atoms = 58  # 原子数量
num_features = 58
total_iter = 0

step = range(30)
Pair = True

model = DeepATC(args, input_dim=num_features)
if args.cuda:
    model.cuda()


###设置不同学习率
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

# loss_func = torch.nn.MSELoss()
# loss_func = torch.nn.BCELoss()  # 多标签分类
loss_func = torch.nn.BCEWithLogitsLoss()  # 多标签分类,相当于不用自带了sigmoid函数
# loss_func = torch.nn.CrossEntropyLoss()
model.train()

def adjust_lr(epoch):
    lrate = args.lr * (0.95 ** np.sum(epoch >= np.array(step)))
    for params_group in optimizer.param_groups:
        params_group['lr'] = lrate
    return lrate

for epoch in range(1, args.epoch + 1):
    save1 = r'records/' + str(CROSS_FOLD) + '_GCN_train_records.txt'
    save2 = r'records/' + str(CROSS_FOLD) + '_GCN_eval_records.txt'
    # best_dict = copy.deepcopy(model.state_dict())
    train_records = open(save1, 'a+')
    eval_records = open(save2, 'a+')

    adjust_lr(epoch)

    train_set_results = []
    start_id = 0
    for i in range(len(y_train) // args.batch_size):
        # print([name for name, param in model.named_parameters()])
        ids_batch, Y_batch, start_id = mini_batch(x_train, y_train, start_id, batchsize=args.batch_size)

        # smi_batch = smi_train[i*args.batch_size:(i+1)*args.batch_size]
        X_batch, A_batch, S_batch = data_transfer.id2graph(ids_batch, drug_dict)
        # print(np.array(X_batch).shape, np.array(Y_batch).shape,np.array(A_batch).shape)
        # Y_batch = logP_train[i * args.batch_size:(i + 1) * args.batch_size]
        X_batch, A_batch, S_batch = torch.tensor(X_batch).float(), torch.tensor(A_batch).float(), torch.tensor(S_batch).float()
        Y_batch = np.array(Y_batch).astype(float)
        Y_batch = torch.Tensor(Y_batch).long()

        if args.cuda:
            X_batch, A_batch, S_batch, Y_batch = X_batch.cuda(), A_batch.cuda(), S_batch.cuda(), Y_batch.cuda()

        optimizer.zero_grad()
        # pred_train = model((X_batch, A_batch))  # drug
        pred_train = model((X_batch, A_batch, S_batch))  # Pair=Drug+Signature
        pred_train, Y_batch = pred_train.cpu(), Y_batch.cpu()
        # print("pred_train, Y_train", pred_train.shape, Y_batch.shape)

        # print(pred_train.shape,Y_batch.shape)
        # Y_batch_one_hot = torch.zeros(args.batch_size, 2).scatter_(1, Y_batch.long(), 1)
        loss = loss_func(pred_train, Y_batch.float())
        # Target = torch.tensor(Y_batch_one_hot).long()
        # print(pred_train)
        # corrects = (torch.max(pred_train, 1)[1] == Y_batch.squeeze().long()).sum()
        # print('corrects:',corrects)
        #
        ###多标签分类
        y_true, y_pred = Y_batch.detach().numpy(), pred_train.detach().numpy()
        trainAUC = roc_auc_score(y_true, y_pred, average='samples')

        y_pred = np.where(y_pred > 0.5, 1, 0)
        Train_acc,Train_P,Train_R,Train_F1 = matric(y_true, y_pred) #acc
        y_true1, y_pred2 = y_true.reshape(-1), y_pred.reshape(-1)
        # print(y_true1.shape, y_pred2.shape)
        trainMCC = matthews_corrcoef(y_true1, y_pred2)
        sys.stdout.write('\rBatch[{}] - loss: {:.4f}  AUC:{:.4f} 准确率: {:.4f} 精确率：{:.3f} 召回率{:.3f} F1值{:.4f} MCC:{:.3f}Epoch{}'.format(
            total_iter, loss.item(),trainAUC,Train_acc, Train_P, Train_R, Train_F1, trainMCC, epoch))

        loss.backward()
        train_records.write('Batch${}$loss${:.5f}$AUC${:.4f}$ACC${:.4f}$Pre${:.3f}$Recall${:.3f}$F1${:.4f}$MCC${:.3f}$Epoch${}\n'.format(
            total_iter, loss.item(),trainAUC,Train_acc, Train_P, Train_R, Train_F1, trainMCC, epoch))

        nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=5.0)
        optimizer.step()
        total_iter += 1

        if total_iter % args.test_interval == 0:
            model.eval()
            print("'\n'!!evaluation")
            X_eval, A_eval, S_eval = data_transfer.id2graph(x_validation, drug_dict)
            Y_eval = y_validation
            X_eval, A_eval, S_eval = torch.tensor(X_eval).float(), torch.tensor(A_eval).float(), torch.tensor(S_eval).float()
            Y_eval = np.array(Y_eval).astype(float)
            Y_eval = torch.Tensor(Y_eval).long()
            if args.cuda:
                X_eval, A_eval, S_eval, Y_eval = X_eval.cuda(), A_eval.cuda(), S_eval.cuda(), Y_eval.cuda()
            pred_eval = model((X_eval, A_eval, S_eval))  # Pair
            # print('results:', pred_eval)
            pred_eval, Y_eval = pred_eval.cpu(), Y_eval.cpu()

            # Y_eval = Y_eval.squeeze()
            # Y_eval_one_hot = torch.zeros(len(Y_eval.numpy()), 2).scatter_(1, Y_eval.long(), 1)
            # print(pred_eval.shape,Y_eval.shape)
            # print("pred_eval, Y_eval",pred_eval.shape, Y_eval.shape)
            loss_eval = loss_func(pred_eval, Y_eval.float())
            eval_loss = loss_eval.item()

            # eval_result = (torch.max(pred_eval, 1)[1] == Y_eval).sum()
            pred_eval = torch.sigmoid(pred_eval)
            pred_eval, Y_eval = pred_eval.detach().numpy(), Y_eval.detach().numpy()
            testAUC = roc_auc_score(Y_eval, pred_eval, average='samples')
            pred_score = np.where(pred_eval > 0.5, 1, 0)
            Test_acc, Test_P, Test_R, Test_F1 = matric(Y_eval, pred_score)
            Y_eval2, pred_score2 = Y_eval.reshape(-1), pred_score.reshape(-1)
            testMCC = matthews_corrcoef(Y_eval2,pred_score2)
            print('\nEvaluation - loss: {:.6f} AUC: {:.4f} 准确率: {:.4f} 精确率：{:.4f} 召回率{:.4f} F1值{:.4f} MCC:{:.4f},Epoch:{} '.format(
                eval_loss, testAUC, Test_acc, Test_P, Test_R, Test_F1,testMCC, epoch)
            )

            eval_records.write('loss${:.6f}$AUC${:.4f}$ACC${:.4f}$P${:.4f}$R${:.4f}$F1${:.4f}$MCC${:.4f}$Epoch${}\n'.format(
                eval_loss, testAUC, Test_acc, Test_P, Test_R, Test_F1, testMCC, epoch))

            if testAUC > best_auc:
                best_auc = testAUC
                # best_dict = copy.deepcopy(model.state_dict())
                # params = [param.cpu().detach().numpy() for name, param in model.fc_pred.named_parameters()]
                # for i in params:
                #     # print(i)
                #     f.write(str(i) + '\n')
                # f.write('$' + '\n')
                # dv = pd.DataFrame(feature)
                # dv.to_csv('/home/wxt/PycharmProjects/Sol/results/LogS_model.csv')
                # feature = np.array(params).resize((-1,1))
                # pred_results = np.array(pred_eval).reshape(-1, 1)
                # Y_results = np.array(Y_eval).reshape(-1, 1)
                # results = np.hstack((pred_results, Y_results))
                # df = pd.DataFrame(results, columns=['pred', 'true'])
                columns = ['DrugID', 'A', 'B', 'C', 'D', 'G', 'H', 'J', 'L', 'M', 'N', 'P', 'R', 'S', 'V']
                index = np.array(x_validation).reshape((-1, 1))
                pred_eval = np.hstack((index, pred_eval))
                Y_eval = np.hstack((index, Y_eval))
                df1 = pd.DataFrame(pred_eval,
                                   columns=columns
                                   )
                df2 = pd.DataFrame(Y_eval,
                                   columns=columns
                                   )
                df1.to_csv('records/' + str(CROSS_FOLD) +'_GCN_Pair_y_score.csv')
                df2.to_csv('records/' + str(CROSS_FOLD) +'_GCN_Pair_y_true.csv')

                save_model = True
                if save_model:
                    torch.save(model.state_dict(), 'checkpoint/GCN_ATC_' + str(CROSS_FOLD) +'.pth')
            print('best- Evaluation_ auc: {:.4f} \n'.format(best_auc))
            model.train()
