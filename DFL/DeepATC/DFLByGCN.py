import torch
import torch.nn as nn

import argparse
parser = argparse.ArgumentParser(description='ATC-Drug')
parser.add_argument('-lr', type=float, default=0.005, help='学习率')
parser.add_argument('-lamda', type=float, default=0.0005, help='学习率')
parser.add_argument('-dp_rate', type=float, default=0.4, help='dropout')
parser.add_argument('-batch_size', type=int, default=256)
parser.add_argument('-epoch', type=int, default=200)
parser.add_argument('-embed_in', type=int, default=700)  # 700, 978
parser.add_argument('-hidden_size', type=int, default=64, help='第一层')
parser.add_argument('-hidden_size2', type=int, default=256, help='第二层')
parser.add_argument('-num_layer', type=int, default=2, help='GCN的层数')
parser.add_argument('-num_class', type=int, default=14, help='lstm stack的层数')
parser.add_argument('-cuda', type=bool, default=True)
parser.add_argument('-using_sc', type=str, default="sc")
parser.add_argument('-log-interval', type=int, default=1, help='经过多少iteration记录一次训练状态')
parser.add_argument('-test-interval', type=int, default=64, help='经过多少iteration对验证集进行测试')
parser.add_argument('-early-stopping', type=int, default=1000, help='早停时迭代的次数')
parser.add_argument('-save-best', type=bool, default=True, help='当得到更好的准确度是否要保存')
parser.add_argument('-save-dir', type=str, default='model_dir', help='存储训练模型位置')
args = parser.parse_args()



class Skip_Connection(nn.Module):
    def __init__(self,input_dim,output_dim,act):
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
    def __init__(self,input_dim,output_dim,act):
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

class HNE(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(HNE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = int(1024)
        self.output_dim = output_dim
        self.act = nn.LeakyReLU()
        # self.embed_layer = nn.Embedding(input_dim, output_dim, padding_idx=0)
        self.fc = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)
        self.bn1 = nn.BatchNorm1d(self.hidden_dim)
        self.bn2 = nn.BatchNorm1d(self.output_dim)
        self.dp = nn.Dropout(0.3)
    def forward(self, x):
        # x = torch.unsqueeze(x,1)
        # x = self.embed_layer(x)
        x = self.fc(x)
        x = self.act(x)
        x = self.dp(x)
        # x = self.bn1(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.dp(x)
        return x

class FA(nn.Module):
    def __init__(self,input_dim1,input_dim2,hidden_dim,act):
        super(FA, self).__init__()
        self.fc_hidden = nn.Linear(input_dim1, hidden_dim)
        self.act = act()
        self.hne = HNE(input_dim2, hidden_dim)
        # self.fa = nn.Sequential(nn.Linear(512, 256), nn.BatchNorm1d(256), nn.Dropout(), nn.ReLU())
        self.attention1 = nn.Sequential(nn.Linear(256, 1), nn.Softmax())
        self.attention2 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(),
                                        nn.Linear(256, 2),
                                        nn.Softmax())

        # self.fc_out = nn.Linear(512, 512)
        # self.slip = torch.nn.Conv1d(in_channels=1,out_channels = 1, kernel_size = 3,stride=1) #Conv1d(in_channels=256,out_channels = 100, kernel_size = 2)
        # self.pool = torch.nn.AvgPool1d(kernel_size=2)

    def forward(self,x1, x2):
        x1 = self.fc_hidden(x1)
        x1 = torch.sum(x1, 1)
        x2 = self.hne(x2)
        x1_ = self.attention1(x1)
        x2_ = self.attention1(x2)
        x1 = torch.mul(x1, x1_)
        x2 = torch.mul(x2, x2_)

        cat1 = torch.cat([x1, x2], dim=1)  # # [128, 512]
        cat2 = self.attention2(cat1)
        # print(x1.shape, x2.shape, cat2[:, 0].shape)
        x1_ = torch.mul(x1, cat2[:, 0].unsqueeze(1))
        x2_ = torch.mul(x2, cat2[:, 1].unsqueeze(1))
        x = x1+x2+x1_+x2_

        # x = self.fa(x)
        return x


class DeepATC(nn.Module):
    def __init__(self, args, input_dim):
        super(DeepATC, self).__init__()
        self.args = args
        self.graph_pre = Graph_Conv(input_dim, args.hidden_size, nn.LeakyReLU, args.using_sc)
        layer_conv = [Graph_Conv(args.hidden_size, args.hidden_size, nn.LeakyReLU, args.using_sc) for i in
                      range(args.num_layer)]
        self.layers = torch.nn.Sequential(*layer_conv)
        self.FA = FA(args.hidden_size, args.embed_in, args.hidden_size2, torch.nn.ReLU)
        self.fc_h1 = nn.Sequential(nn.Linear(256, args.hidden_size2), nn.BatchNorm1d(args.hidden_size2), nn.Dropout(args.dp_rate))  # args.hidden_size2
        self.fc_pred = nn.Linear(args.hidden_size2, args.num_class)

    def forward(self, input):
        x1, A, x2 = input
        x1, A = self.graph_pre((x1, A))
        x1, A = self.layers((x1, A))
        x = self.FA(x1, x2)
        self.featuremap1 = x.detach()
        # x = self.fc_h2(x)
        Y_pred = self.fc_pred(x).float()
        # print(Y_pred)
        return Y_pred