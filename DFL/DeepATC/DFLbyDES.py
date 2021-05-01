import torch
import torch.nn as nn

import argparse
parser = argparse.ArgumentParser(description='ATC-Drug')
parser.add_argument('-lr', type=float, default=0.001, help='学习率')
parser.add_argument('-lamda', type=float, default=0.0005, help='学习率')
parser.add_argument('-dp_rate', type=float, default=0.4, help='dropout')
parser.add_argument('-batch_size', type=int, default=256)
parser.add_argument('-epoch', type=int, default=200)
parser.add_argument('-embed_in', type=int, default=167)  # 700, 978, 167
parser.add_argument('-hidden_size', type=int, default=64, help='第一层')
parser.add_argument('-hidden_size2', type=int, default=256, help='第二层')
parser.add_argument('-num_layer', type=int, default=1, help='lstm stack的层数')
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

class Embedding(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(Embedding, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = int(1024)  # TODO:重点调整的隐藏层
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

class Readout(nn.Module):
    def __init__(self, input_dim1,input_dim2,hidden_dim,act):
        super(Readout, self).__init__()
        self.act=act()
        self.fc_hidden = nn.Linear(input_dim1, hidden_dim)
        self.dropout = nn.Dropout(args.dp_rate) #TODO: Dropout

        self.bn = nn.BatchNorm1d(256)  # TODO:
        self.hneEmbedding = Embedding(100, 512)  # hidden_dim
        self.fpsEmbedding = Embedding(167, 512)
        # self.fc_out = nn.Linear(512, 512)
        # self.slip = torch.nn.Conv1d(in_channels=1,out_channels = 1, kernel_size = 3,stride=1) #Conv1d(in_channels=256,out_channels = 100, kernel_size = 2)
        self.pool = torch.nn.AvgPool1d(kernel_size=2)
    def forward(self,x1, x2):

        x1 = self.fpsEmbedding(x1)
        x2 = self.hneEmbedding(x2)
        x = torch.cat([x1, x2], dim=1)  # #[128, 256+ 256]

        x = torch.unsqueeze(x2, 1)  #[128, 1, 512]
        #
        # x = self.slip(x)
        x = self.pool(x)  #[128, 1, 256]

        # print('Readout----', x.shape)
        # x = self.slip(x)
        # x = self.fc_out(x)

        x = torch.squeeze(x, 1)
        x = self.act(x)  # TODO: batchnorm + Dropout
        x = self.bn(x)
        # print('Readout',x.shape,x)
        return x


class DeepATC(nn.Module):
    def __init__(self, args, input_dim):
        super(DeepATC, self).__init__()
        self.args = args
        self.graph_pre = Graph_Conv(input_dim, args.hidden_size, nn.LeakyReLU, args.using_sc)
        layer_conv = [Graph_Conv(args.hidden_size, args.hidden_size, nn.LeakyReLU, args.using_sc) for i in
                      range(args.num_layer)]
        self.layers = torch.nn.Sequential(*layer_conv)
        self.readout = Readout(args.hidden_size, args.embed_in, args.hidden_size2, torch.nn.ReLU)
        self.fc_h1 = nn.Linear(256, args.hidden_size2)  # args.hidden_size2
        # self.fc_h2 = nn.Linear(args.hidden_size2, args.hidden_size2)
        self.bn = nn.BatchNorm1d(256)  # TODO: BatchNorm1  #256
        self.dropout = nn.Dropout(args.dp_rate)  # TODO: Dropout

        self.fc_pred = nn.Linear(args.hidden_size2, args.num_class)
    def forward(self, inputs):
        x1, x2 = inputs
        # x1, A = self.graph_pre((x1, A))
        # x1, A = self.layers((x1, A))

        x = self.readout(x1, x2)
        x = self.bn(self.fc_h1(x))  # TODO: Dropout
        self.featuremap1 = x.detach()
        # x = self.fc_h2(x)
        Y_pred = self.fc_pred(x).float()
        # print(Y_pred)
        return Y_pred