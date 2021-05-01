import argparse

# Model Hyper-parameters
parser = argparse.ArgumentParser(description='DFL framework for DTI prediction')
parser.add_argument('-lr', type=float, default=0.01, help='learning rate')
parser.add_argument('-lamda', type=float, default=0.0005, help='decay-rate')
parser.add_argument('-batch_size', type=int, default=128, help='batch size')
parser.add_argument('-epoch', type=int, default=100, help='epoch')
parser.add_argument('-dp_rate', type=float, default=0.3, help='dropout rate')
parser.add_argument('-num_class', type=int, default=1, help='number of neurons in output layer')
parser.add_argument('-cuda', type=bool, default=True, help='Accelerating with CUDA')

# GCN Module Hyper-parameters
parser.add_argument('-num_features', type=int, default=58, help='number of molecule features')
parser.add_argument('-num_gcn_layer', type=int, default=1, help='number of graph convolution network layers')
parser.add_argument('-gcn_hidden_dim', type=int, default=64, help='hidden layer size of GCN Module')
parser.add_argument('-gcn_output_dim', type=int, default=256, help='output layer size of GCN Module')
parser.add_argument('-using_sc', type=str, default="gsc")  # sc
# Embedding Net Hyper-parameters
parser.add_argument('-drug_dim', type=int, default=100, help='embedding dimension of drug vector')
parser.add_argument('-protein_dim', type=int, default=400, help='embedding dimension of protein vector')
parser.add_argument('-emb_hidden_size', type=int, default=512, help='number of neurons in the EmbeddingNet hidden layer')
parser.add_argument('-drug_output_dim', type=int, default=256, help='number of neurons in drug Embedding output layer')
parser.add_argument('-protein_output_dim', type=int, default=512, help='number of neurons in protein Embedding output layer')

# FA Module Hyper-parameters
# FA inputs, hidden, outputs
parser.add_argument('-fa_input_dim', type=int, default=1024, help='input layer size of FA Module = gcn_hidden_dim2 + drug_output_dim + protein_output_dim')
parser.add_argument('-fa_hidden_dim', type=int, default=512, help='hidden layer size of FA Module')
parser.add_argument('-fa_output_dim', type=int, default=512, help='output layer size of FA Module')

# model training
parser.add_argument('-log-interval', type=int, default=1, help='经过多少iteration记录一次训练状态')
parser.add_argument('-test-interval', type=int, default=64, help='经过多少iteration对验证集进行测试')
