import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
parser = argparse.ArgumentParser(description='DTI')
parser.add_argument('-lr', type=float, default=0.005, help='学习率')
parser.add_argument('-lamda', type=float, default=0.0005, help='学习率')
parser.add_argument('-dp_rate', type=float, default=0.6, help='dropout')
parser.add_argument('-batch_size', type=int, default=128)
parser.add_argument('-epoch', type=int, default=200)
parser.add_argument('-embed_in', type=int, default=700)  # 700, 978
parser.add_argument('-hidden_size', type=int, default=64, help='第一层')
parser.add_argument('-hidden_size2', type=int, default=256, help='第二层')
parser.add_argument('-num_layer', type=int, default=1, help='GCN的层数')
parser.add_argument('-cuda', type=bool, default=True)
parser.add_argument('-using_sc', type=str, default="sc")
parser.add_argument('-log-interval', type=int, default=1, help='经过多少iteration记录一次训练状态')
parser.add_argument('-test-interval', type=int, default=64, help='经过多少iteration对验证集进行测试')
parser.add_argument('-early-stopping', type=int, default=1000, help='早停时迭代的次数')
parser.add_argument('-save-best', type=bool, default=True, help='当得到更好的准确度是否要保存')
parser.add_argument('-save-dir', type=str, default='model_dir', help='存储训练模型位置')
parser.add_argument('-smiles', type=str, default=1000, help='molecule_representation')  # 'graph, smiles, descriptor'
parser.add_argument('-embed_num', type=int, default=22, help='token词表长度')
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
parser.add_argument('-class_num', type=int, default=1, help='类别数')
parser.add_argument('-gru_dp', type=int, default=0.5, help='dropout')
parser.add_argument('-dropout', type=int, default=0.7, help='dropout')

args = parser.parse_args()


class Skip_Connection(nn.Module):
    def __init__(self, input_dim,output_dim,act):
        super(Skip_Connection, self).__init__()
        self.act = act()
        self.input_dim=input_dim
        self.output_dim=output_dim
        if input_dim!=output_dim:
            self.fc_1=nn.Linear(input_dim, output_dim)

    def forward(self,input):#input=[X,new_X]
        x,new_X=input
        if self.input_dim!=self.output_dim:
            out=self.fc_1(x)
            x=self.act(out+new_X)
        else:
            x = self.act(x + new_X)
        return x

class Gated_Skip_Connection(nn.Module):
    def __init__(self, input_dim,output_dim,act):
        super(Gated_Skip_Connection, self).__init__()

        self.act = act()
        self.input_dim = input_dim
        self.output_dim = output_dim
        if input_dim != output_dim:
            self.fc_1 = nn.Linear(input_dim, output_dim)
        self.gated_X1=nn.Linear(self.output_dim, self.output_dim)
        self.gated_X2=nn.Linear(self.output_dim, self.output_dim)

    def forward(self,input):
        x,x_new=input
        if self.input_dim != self.output_dim:
            x = self.fc_1(x)
        gate_coefficient = torch.sigmoid(self.gated_X1(x)+self.gated_X2(x_new))
        x=x_new.mul(gate_coefficient)+x.mul((1.0-gate_coefficient))
        return x

class Graph_Conv(nn.Module):
    def __init__(self, input_dim, hidden_dim, act, using_sc):
        super(Graph_Conv, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.fc_hidden = nn.Linear(self.input_dim, self.hidden_dim)
        self.act = act()
        self.using_sc = using_sc

        if (using_sc == 'sc'):
            self.skip_connection = Skip_Connection(self.input_dim, self.hidden_dim, act)
        if (using_sc == 'gsc'):
            self.skip_connection = Gated_Skip_Connection(self.input_dim, self.hidden_dim, act)
        # if (using_sc=="no"):
        #     output_X = act(output_X)
        # self.bn = nn.BatchNorm1d(50)

    def forward(self, inputs):

        x, A = inputs
        # print('bug----', x.shape)
        x_new = self.fc_hidden(x)  # [Batch,N,H]
        x_new = torch.bmm(A, x_new)
        # print(x_new.shape)
        # print("x:", x.shape, "A:", A.shape)
        if self.using_sc == "no":
            x = self.act(x_new)
        else:
            x = self.skip_connection((x, x_new))
        return (x, A)


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
        x = embed.view(len(input), embed.size(1), -1)
        # gru
        gru_out, _ = self.bigru(x)
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

    def __init__(self, input_dim, output_dim):
        super(HNE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = int(256)  # TODO:重点调整的隐藏层
        self.output_dim = output_dim
        self.act = nn.LeakyReLU()
        # self.embed_layer = nn.Embedding(input_dim, output_dim, padding_idx=0)
        self.fc = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)
        self.bn1 = nn.BatchNorm1d(self.hidden_dim)
        self.bn2 = nn.BatchNorm1d(self.output_dim)
        self.dp = nn.Dropout(0.4)
    def forward(self, x):
        # x = torch.unsqueeze(x,1)
        # x = self.embed_layer(x)
        x = self.fc(x)
        x = self.act(x)
        x = self.dp(x)
        # x = self.bn1(x)
        # x = self.fc2(x)
        # x = self.act(x)
        # x = self.dp(x)
        return x

# class MAF(nn.Module):
#     def __init__(self,input_dim1,input_dim2,hidden_dim,act):
#         super(MAF, self).__init__()
#         self.fc_hidden = nn.Linear(input_dim1, hidden_dim)
#         self.act = act()
#         self.hne = HNE(input_dim2, hidden_dim)
#         # self.fa = nn.Sequential(nn.Linear(512, 256), nn.BatchNorm1d(256), nn.Dropout(), nn.ReLU())
#         self.attention1 = nn.Sequential(nn.Linear(256, 1), nn.Softmax())
#         self.attention2 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(),
#                                         nn.Linear(256, 2),
#                                         nn.Softmax())
#
#         # self.fc_out = nn.Linear(512, 512)
#         # self.slip = torch.nn.Conv1d(in_channels=1,out_channels = 1, kernel_size = 3,stride=1) #Conv1d(in_channels=256,out_channels = 100, kernel_size = 2)
#         # self.pool = torch.nn.AvgPool1d(kernel_size=2)
#
#     def forward(self, x1, x2):
#         x1 = self.fc_hidden(x1)
#         x1 = torch.sum(x1, 1)
#         x2 = self.hne(x2)
#         x1_ = self.attention1(x1)
#         x2_ = self.attention1(x2)
#         x1 = torch.mul(x1, x1_)
#         x2 = torch.mul(x2, x2_)
#
#         cat1 = torch.cat([x1, x2], dim=1)  # # [128, 512]
#         cat2 = self.attention2(cat1)
#         # print(x1.shape, x2.shape, cat2[:, 0].shape)
#         x1_ = torch.mul(x1, cat2[:, 0].unsqueeze(1))
#         x2_ = torch.mul(x2, cat2[:, 1].unsqueeze(1))
#         x = x1+x2+x1_+x2_
#
#         # x = self.fa(x)
#         return x



class DTI(nn.Module):
    def __init__(self, args, input_dim):
        super(DTI, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.num_layer = args.num_layer

        self.graph_pre = Graph_Conv(input_dim, self.hidden_size, nn.LeakyReLU, args.using_sc)
        layer_conv = [Graph_Conv(args.hidden_size, args.hidden_size, nn.LeakyReLU, args.using_sc) for i in
                      range(self.num_layer)]
        self.layers = torch.nn.Sequential(*layer_conv)

        self.gru = BiGRU(args)
        self.drug_emb = HNE(100, 256)
        self.protein_emb = HNE(400, 256)
        self.gcn_hidden = nn.Sequential(nn.Linear(64, 256), nn.ReLU())
        self.fc_h1 = nn.Sequential(nn.Linear(768, 512),
                                   nn.BatchNorm1d(512),
                                   nn.Dropout(args.dp_rate),
                                   nn.ReLU(),
                                   nn.Linear(512, 256),
                                   nn.BatchNorm1d(256),
                                   nn.ReLU()
                                   )  # args.hidden_size2
        self.fc_pred = nn.Linear(256, args.class_num)

    def forward(self, input):
        dF, dA, dv, ps, pv = input
        dv = self.drug_emb(dv)
        dF, dA = self.graph_pre((dF, dA))
        dF, dA = self.layers((dF, dA))
        dF = torch.sum(dF, 1)
        dF = self.gcn_hidden(dF)
        p_seq = self.gru(ps)

        p_vec = self.protein_emb(pv)
        # print(dF.shape, dv.shape)
        drug = torch.cat([dF, dv], dim=1)
        # protein = torch.cat([p_seq, p_vec], dim=1)
        x = torch.cat([drug, p_seq], dim=1)

        x = self.fc_h1(x)

        self.featuremap1 = x.detach()
        # x = self.fc_h2(x)
        Y_pred = self.fc_pred(x).float()
        # print(Y_pred.shape)
        return Y_pred


# class Word4ATC(nn.Module):
#     def __init__(self, args):
#         super(Word4ATC, self).__init__()
#         self.fa_input_dim = args.fa_cat_dim
#         self.fa_hidden_dim = args.fa_hidden_dim
#         self.fc_hidden_dim = args.fc_hidden_dim
#         self.BiGRU = BiGRU(args)
#         self.HNE = HNE(args)
#         self.FA = nn.Sequential(nn.Linear(self.fa_input_dim, self.fa_hidden_dim), nn.Dropout(), nn.ReLU())
#
#         self.num_layers = args.class_num
#         self.predictor = nn.Sequential(nn.Linear(self.fa_hidden_dim, self.fc_hidden_dim), nn.Dropout(), nn.ReLU(),
#                                        nn.Linear(self.fc_hidden_dim, self.num_layers))
#         # self.predictor = nn.Sequential(nn.Linear(self.fa_hidden_dim, self.num_layers))
#
#     def forward(self, input):
#         smi, net_vector = input
#         gru_out = self.BiGRU(smi)
#         hne_out = self.HNE(net_vector)
#
#         fa_in = torch.cat([gru_out, hne_out], dim=1)
#         fa_out = self.FA(fa_in)
#         y = self.predictor(fa_out)
#         logit = y
#         return logit