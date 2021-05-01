import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch
import sys
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from utils import DataLoader, DataTransfer, mini_batch, matric
np.seterr(invalid='ignore')

import argparse
parser = argparse.ArgumentParser(description='ATC-Drug')
parser.add_argument('-lr', type=float, default=0.005, help='学习率')
parser.add_argument('-lamda', type=float, default=0.0005, help='学习率')
parser.add_argument('-batch_size', type=int, default=256)
parser.add_argument('-epoch', type=int, default=200)
parser.add_argument('-smiles', type=str, default=1000, help='molecule_representation')  # 'graph, smiles, descriptor'
parser.add_argument('-embed_num', type=int, default=60, help='token词表长度')
parser.add_argument('-embed_dim', type=int, default=200, help='词向量长度')
parser.add_argument('-gru_hidden_dim', type=int, default=300, help='隐藏层')
parser.add_argument('-gru_hidden_dim2', type=int, default=256, help='隐藏层')
parser.add_argument('-gru_num_layers', type=int, default=1, help='lstm_num_layers')
parser.add_argument('-hne_input_dim', type=int, default=700, help='HNE initial dim')
parser.add_argument('-hne_hidden_dim', type=int, default=512, help='HNE hidden dim')
parser.add_argument('-hne_output_dim', type=int, default=256, help='HNE output dim')
parser.add_argument('-fa_cat_dim', type=int, default=512, help='HNE cat dim')
parser.add_argument('-fa_hidden_dim', type=int, default=256, help='HNE output dim')
parser.add_argument('-fc_hidden_dim', type=int, default=256, help='HNE output dim')
parser.add_argument('-class_num', type=int, default=14, help='类别数')
parser.add_argument('-gru_dp', type=int, default=0.5, help='dropout')
parser.add_argument('-dropout', type=int, default=0.7, help='dropout')
parser.add_argument('-cuda', type=bool, default=False)
parser.add_argument('-test-interval', type=int, default=100, help='经过多少iteration对验证集进行测试')


args = parser.parse_args()

class BiGRU(nn.Module):

    def __init__(self, args):
        super(BiGRU, self).__init__()
        self.args = args
        self.gru_hidden_dim = args.gru_hidden_dim
        self.gru_hidden_dim2 = args.gru_hidden_dim2
        self.gru_num_layers = args.gru_num_layers
        V = args.embed_num
        D = args.embed_dim

        self.embed = nn.Embedding(V, D, padding_idx=0)
        # gru
        self.bigru = nn.GRU(D, self.gru_hidden_dim, dropout=args.dropout, num_layers=self.gru_num_layers, bidirectional=True)
        # linear
        self.hidden2fc = nn.Sequential(nn.Linear(self.gru_hidden_dim * 2, args.gru_hidden_dim2), nn.Dropout(args.gru_dp), nn.ReLU())
        # self.fc2label = nn.Sequential(nn.Linear(args.fc_hidden_dim, C))
        #  dropout
        self.dropout = nn.Dropout(args.gru_dp)

    def forward(self, input):
        embed = self.embed(input)
        embed = self.dropout(embed)
        input = embed.view(len(input), embed.size(1), -1)
        # gru
        gru_out, _ = self.bigru(input)
        # gru_out = torch.transpose(gru_out, 0, 1)
        gru_out = torch.transpose(gru_out, 1, 2)
        # pooling
        # gru_out = F.tanh(gru_out)
        gru_out = F.max_pool1d(gru_out, gru_out.shape[2]).squeeze(2)
        gru_out = torch.tanh(gru_out)
        # linear
        fc_out = self.hidden2fc(gru_out)
        # y = self.fc2label(fc_in)
        # logit = y
        # print('logit', logit.shape)
        return fc_out

class HNE(nn.Module):
    def __init__(self, args):
        super(HNE, self).__init__()
        self.input_dim = args.hne_input_dim
        self.hidden_dim = args.hne_hidden_dim  # TODO:重点调整的隐藏层
        self.output_dim = args.hne_output_dim
        self.act = nn.LeakyReLU()
        # self.embed_layer = nn.Embedding(input_dim, output_dim, padding_idx=0)
        self.fc = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)
        self.bn1 = nn.BatchNorm1d(self.hidden_dim)
        self.bn2 = nn.BatchNorm1d(self.output_dim)
        self.dp = nn.Dropout(0.5)

    def forward(self, input):
        # x = torch.unsqueeze(x,1)
        # x = self.embed_layer(x)
        x = self.fc(input)
        x = self.act(x)
        x = self.dp(x)
        # x = self.bn1(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.dp(x)
        return x

class Word4ATC(nn.Module):
    def __init__(self, args):
        super(Word4ATC, self).__init__()
        self.fa_input_dim = args.fa_cat_dim
        self.fa_hidden_dim = args.fa_hidden_dim
        self.fc_hidden_dim = args.fc_hidden_dim
        self.BiGRU = BiGRU(args)
        self.HNE = HNE(args)
        self.FA = nn.Sequential(nn.Linear(self.fa_input_dim, self.fa_hidden_dim), nn.Dropout(), nn.ReLU())

        self.num_layers = args.class_num
        self.predictor = nn.Sequential(nn.Linear(self.fa_hidden_dim, self.fc_hidden_dim), nn.Dropout(), nn.ReLU(),
                                       nn.Linear(self.fc_hidden_dim, self.num_layers))
        # self.predictor = nn.Sequential(nn.Linear(self.fa_hidden_dim, self.num_layers))

    def forward(self, input):
        smi, net_vector = input
        gru_out = self.BiGRU(smi)
        hne_out = self.HNE(net_vector)
        fa_in = torch.cat([gru_out, hne_out], dim=1)
        fa_out = self.FA(fa_in)
        y = self.predictor(fa_out)
        logit = y
        return logit


drug_data = 'train_data/drugATC.json'  # {'drugID': {'vector': '0.1,0.2,...', 'SMILES': 'CC', 'label': '[1, 0, ..]'}}
cv_json = 'train_data/CV_ATC.json'  # cross validation data index
data_loader = DataLoader(drug_data, cv_json)
drug_dict = data_loader.data_dict
CROSS_FOLD = 9

x_train, x_validation, y_train, y_validation = data_loader.split(CROSS_FOLD, CV=True, test_size=0.3)

data_transfer = DataTransfer()


print('Train:', len(y_train))
print('Test:', len(y_validation), x_validation[:5])

#  init param
best_auc = 0
avg_loss = 0

TOTAL_ITER = 0

step = range(30)
Pair = True

model = Word4ATC(args)
if args.cuda:
    model.cuda()

###设置不同学习率
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

# loss_func = torch.nn.MSELoss()
# loss_func = torch.nn.BCELoss()#多标签分类
loss_func = torch.nn.BCEWithLogitsLoss()  #多标签分类,相当于不用自带了sigmoid函数
# loss_func = torch.nn.CrossEntropyLoss()
model.train()

def adjust_lr(epoch):
    lrate = args.lr * (0.95 ** np.sum(epoch >= np.array(step)))
    for params_group in optimizer.param_groups:
        params_group['lr'] = lrate
    return lrate

for epoch in range(1, args.epoch + 1):
    save1 = r'records/'+str(CROSS_FOLD)+'_GRU_train_records.txt'
    save2 = r'records/'+str(CROSS_FOLD)+'_GRU_eval_records.txt'
    # best_dict = copy.deepcopy(model.state_dict())
    train_records = open(save1, 'a+')
    eval_records = open(save2, 'a+')


    adjust_lr(epoch)

    train_set_results = []
    start_id = 0
    for i in range(len(y_train) // args.batch_size):
        # print([name for name, param in model.named_parameters()])
        ids_batch, _, start_id = mini_batch(x_train, y_train, start_id, batchsize=args.batch_size)

        # smi_batch = smi_train[i*args.batch_size:(i+1)*args.batch_size]
        W_batch, V_batch, Y_batch = data_transfer.id2word(ids_batch, drug_dict)
        W_batch, V_batch = torch.tensor(W_batch).long(), torch.tensor(V_batch).float()
        Y_batch = np.array(Y_batch).astype(float)
        Y_batch = torch.Tensor(Y_batch).long()

        if args.cuda:
            W_batch, V_batch, Y_batch = W_batch.cuda(), V_batch.cuda(), Y_batch.cuda()

        optimizer.zero_grad()
        # pred_train = model((X_batch, A_batch))  # drug
        pred_train = model((W_batch, V_batch))  # Pair=Drug+Signature
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
        Train_acc, Train_P, Train_R, Train_F1 = matric(y_true, y_pred) #acc
        y_true1, y_pred2 = y_true.reshape(-1), y_pred.reshape(-1)
        # print(y_true1.shape, y_pred2.shape)
        trainMCC = matthews_corrcoef(y_true1, y_pred2)
        sys.stdout.write('\rBatch[{}] - loss: {:.4f}  AUC:{:.4f} 准确率: {:.4f} 精确率：{:.3f} 召回率{:.3f} F1值{:.4f} MCC:{:.3f}Epoch{}'.format(
            TOTAL_ITER, loss.item(),trainAUC,Train_acc, Train_P, Train_R, Train_F1, trainMCC, epoch))

        train_records.write(
            'Batch:{}$loss:{:.5f}$AUC:{:.4f}$ACC:{:.4f}$AP:{:.3f}$RE:{:.3f}$F1:{:.4f}$MCC:{:.3f}$Epoch:{}\n'.format(
                TOTAL_ITER, loss.item(), trainAUC, Train_acc, Train_P, Train_R, Train_F1, trainMCC, epoch))
        loss.backward()
        train_records.write(str(loss.item()) + '$' + str(Train_acc) + '\n')
        nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=5.0)
        optimizer.step()
        TOTAL_ITER += 1

        if TOTAL_ITER % args.test_interval == 0:
            model.eval()
            print("'\n'!!evaluation")
            W_eval, V_eval, Y_eval = data_transfer.id2word(x_validation, drug_dict)
            W_eval, V_eval, Y_eval = torch.tensor(W_eval).long(), torch.tensor(V_eval).float(), torch.tensor(Y_eval).float()
            Y_eval = np.array(Y_eval).astype(float)
            Y_eval = torch.Tensor(Y_eval).long()
            if args.cuda:
                W_eval, V_eval, Y_eval = W_eval.cuda(), V_eval.cuda(), Y_eval.cuda()
            pred_eval = model((W_eval, V_eval))
            # print('results:',pred_eval)
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
            Test_acc, Test_P, Testn_R, Test_F1 = matric(Y_eval, pred_score)
            Y_eval2, pred_score2 = Y_eval.reshape(-1), pred_score.reshape(-1)
            testMCC = matthews_corrcoef(Y_eval2,pred_score2)
            print('\nEvaluation - loss: {:.6f} AUC: {:.4f} 准确率: {:.4f} 精确率：{:.3f} 召回率{:.3f} F1值{:.4f} MCC:{:.3f},Epoch:{} '.format(
                eval_loss, testAUC, Test_acc, Test_P, Testn_R, Test_F1,testMCC, epoch)
            )
            eval_records.write(
                'loss:{:.5f}$AUC:{:.4f}$ACC:{:.4f}$ACC:{:.3f}$RE:{:.3f}$F1:{:.4f}$MCC:{:.3f}$Epoch:{}\n'.format(
                    eval_loss, testAUC, Test_acc, Test_P, Testn_R, Test_F1, testMCC, epoch))

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
                df1.to_csv('records/'+str(CROSS_FOLD)+'_GRU_Pair_y_score.csv')
                df2.to_csv('records/'+str(CROSS_FOLD)+'_GRU_Pair_y_true.csv')

                save_model = True
                if save_model:
                    torch.save(model.state_dict(), 'checkpoint/GRU_ATC_'+str(CROSS_FOLD)+'.pth')
            print('best- Evaluation_ auc: {:.4f} \n'.format(best_auc))
            model.train()