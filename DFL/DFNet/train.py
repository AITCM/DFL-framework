import torch
import numpy as np
from DTIbyGRU import DTI, parser

from sklearn.metrics import roc_auc_score, matthews_corrcoef
import copy
from utils import DataLoader, DataTransfer, mini_batch, matric
import sys
import torch.nn as nn
import json
import pandas as pd

args = parser.parse_args()

drug_target_data = 'train_data/Data-python-4.28.json'
cv_json = 'train_data/CV_DTI.json'
dataLoader = DataLoader(drug_target_data, cv_json)
data_dict = dataLoader.data_dict
data_transfer = DataTransfer()
CROSS_FOLD = 2
x_train, x_validation, y_train, y_validation = dataLoader.split(CROSS_FOLD, CV=True, test_size=0.3)
print('Train:', len(x_train))
print('Test:', len(x_validation))

num_features = 58

def adjust_lr(epoch):
    lrate = args.lr * (0.95 ** np.sum(epoch >= np.array(STEP)))
    for params_group in optimizer.param_groups:
        params_group['lr'] = lrate
    return lrate

if __name__ == '__main__':
    #  initial parameters
    pathW1 = r'records/trainRecords-4.7.txt'
    pathW2 = r'records/evalRecords-4.7.txt'


    BEST_AUC = 0
    TOTAL_ITER = 0
    STEP = 30  # range(30)
    # data set

    # model
    model = DTI(args, input_dim=num_features)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    loss_func = torch.nn.BCEWithLogitsLoss()
    if args.cuda:
        model.cuda()
    model.train()

    # model training
    for epoch in range(1, args.epoch + 1):
        adjust_lr(epoch)
        record1 = open(pathW1, 'a+')
        record2 = open(pathW2, 'a+')
        start_id = 0
        for i in range(len(y_train) // args.batch_size):
            ids_batch, _, start_id = mini_batch(x_train, y_train, start_id, batchsize=args.batch_size)
            df_batch, da_batch, dv_batch, ps_batch, pv_batch, Y_batch = data_transfer.id2dti(ids_batch, data_dict)
            df_batch = torch.tensor(df_batch).float()
            da_batch = torch.tensor(da_batch).float()
            dv_batch = torch.tensor(dv_batch).float()
            ps_batch = torch.tensor(ps_batch).long()
            pv_batch = torch.tensor(pv_batch).float()
            Y_batch = np.array(Y_batch).astype(float)
            Y_batch = torch.Tensor(Y_batch).long()

            if args.cuda:
                df_batch, da_batch, dv_batch, ps_batch, pv_batch = \
                    df_batch.cuda(), da_batch.cuda(), dv_batch.cuda(), ps_batch.cuda(), pv_batch.cuda()
                Y_batch = Y_batch.cuda()

            optimizer.zero_grad()

            trainPred = model((df_batch, da_batch, dv_batch, ps_batch, pv_batch))
            trainPred, Y_batch = trainPred.cpu(), Y_batch.cpu()
            Y_batch = Y_batch.unsqueeze(1)

            loss = loss_func(trainPred, Y_batch.float())

            trainY, trainPred = Y_batch.detach().numpy(),trainPred.detach().numpy()
            trainAUC = roc_auc_score(trainY, trainPred)

            trainTarget = np.where(trainPred > 0.5, 1, 0)
            trainMCC = matthews_corrcoef(trainY, trainTarget)
            trainACC, trainP, trainR, trainF1 = matric(trainY, trainTarget)
            sys.stdout.write(
                '\rbatch:{} loss:{:.4f} AUC:{:.3f} ACC:{:.4f} precision{:.3f} recall:{:.3f} F1{:.3f} MCC:{:.3f} Epoch{}'.format(
                    TOTAL_ITER, loss.item(), trainAUC, trainACC, trainP, trainR, trainF1, trainMCC, epoch))

            loss.backward()
            nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=5.0)
            optimizer.step()
            record1.write('batch:{}$loss:{:.4f}$AUC:{:.3f}$ACC:{:.4f}$precision:{:.3f}$recall:{:.3f}$F1{:.3f}$MCC:{:.3f}$Epoch{}\n'.format(
                    TOTAL_ITER, loss.item(), trainAUC, trainACC, trainP, trainR, trainF1, trainMCC, epoch))

            TOTAL_ITER += 1
            if TOTAL_ITER % args.test_interval == 0:
                model.eval()
                df_eval, da_eval, dv_eval, ps_eval, pv_eval, testY = data_transfer.id2dti(x_validation, data_dict)

                df_eval, da_eval, dv_eval, ps_eval, pv_eval = \
                    torch.tensor(df_eval).float(), torch.tensor(da_eval).float(), torch.tensor(dv_eval).float(),\
                    torch.tensor(ps_eval).long(), torch.tensor(pv_eval).float(),

                testY = np.array(testY).astype(float)
                testY = torch.Tensor(testY).long()

                if args.cuda:
                    df_eval, da_eval, dv_eval, ps_eval, pv_eval = \
                        df_eval.cuda(), da_eval.cuda(), dv_eval.cuda(), ps_eval.cuda(), pv_eval.cuda()
                    testY = testY.cuda()

                testPred = model((df_eval, da_eval, dv_eval, ps_eval, pv_eval))
                testPred, testY = testPred.cpu(), testY.cpu()
                testY = testY.unsqueeze(1)
                loss_eval = loss_func(testPred, testY.float())
                eval_loss = loss_eval.item()

                testPred, testY = testPred.detach().numpy(), testY.detach().numpy()

                testAUC = roc_auc_score(testY, testPred)

                testTarget = np.where(testPred > 0.5, 1, 0)
                testMCC = matthews_corrcoef(testY, testTarget)
                testACC, testP, testR, testF1 = matric(testY, testTarget)
                print('\nEval - loss: {:.4f} AUC:{:.3f} ACC: {:.4f} P：{:.3f} R{:.3f} F1值{:.4f} MCC{:.4f} Epoch:{} '.format(
                    eval_loss, testAUC, testACC, testP, testR, testF1, testMCC, epoch))

                # record2.write(
                #     str(loss_eval.item()) + '$' + str(CrossFold) + '$' + str(testAUC) + '\n')
                record2.write('CV:{} loss:{:.4f} AUC:{:.3f} ACC:{:.4f} precision:{:.3f} recall{:.3f} F1{:.4f} MCC{:.4f} Epoch:{}\n'.format(
                    CROSS_FOLD, eval_loss, testAUC, testACC, testP, testR, testF1, testMCC, epoch))


                if testAUC > BEST_AUC:
                    BEST_AUC = testAUC
                    best_dict = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), 'checkpoint/DFNet.pth')

                    df = np.hstack((testY, testPred))
                    df = pd.DataFrame(df, columns=['True', 'Pred'])
                    df.to_csv('records/DFNetPrediction.csv')
                print('\n best- eval_AUC: {:.4f}'.format(BEST_AUC))
                model.train()

