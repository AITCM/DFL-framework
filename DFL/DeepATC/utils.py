from rdkit import Chem
import rdkit.Chem.AllChem as AllChem
import rdkit.Chem.rdmolops as rdmolops
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from rdkit.ML.Descriptors import MoleculeDescriptors
import rdkit
import torch



def mini_batch(X_train, Y_train, start_ind, batchsize=32):

    if start_ind+batchsize <= len(Y_train):
        return np.array(X_train[start_ind:start_ind+batchsize]), Y_train[start_ind:start_ind+batchsize], start_ind+batchsize
    else:
        end_ind = (start_ind+batchsize) % len(Y_train)
        X_batch = np.concatenate(([X_train[start_ind:len(Y_train)], X_train[:end_ind]]), axis=0)
        Y_batch = np.concatenate(([Y_train[start_ind:len(Y_train)], Y_train[:end_ind]]), axis=0)
        return X_batch, Y_batch, end_ind

def matric(y_true,y_pred):
    # True Positive:即y_true与y_pred中同时为1的个数
    TP = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 1)))  #
    # TP = np.sum(np.multiply(y_true, y_pred)) #同样可以实现计算TP
    # False Positive:即y_true中为0但是在y_pred中被识别为1的个数
    FP = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 1)))  #
    # False Negative:即y_true中为1但是在y_pred中被识别为0的个数
    FN = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 0)))  #
    # True Negative:即y_true与y_pred中同时为0的个数
    TN = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 0)))  #

    # 根据上面得到的值计算A、P、R、F1
    A = (TP + TN) / (TP + FP + FN + TN)  # y_pred与y_ture中同时为1或0
    P = TP / (TP + FP)  # y_pred中为1的元素同时在y_true中也为1
    R = TP / (TP + FN)  # y_true中为1的元素同时在y_pred中也为1
    F1 = 2 * P * R / (P + R)

    return A, P, R, F1


def read_data(path1, path2):
    smiles = []
    sigs = []  # landmark gene signatures
    labels = []
    # contents = ['DrugbankID', 'SMILES', 'A', 'B', 'C', 'D', 'G', 'H', 'J', 'L', 'M', 'N', 'P', 'R', 'S', 'V']
    contents = ['KEGG DrugID', 'SMILES', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14']
    df1 = pd.read_csv(path1, encoding='gbk')
    df2 = pd.read_csv(path2, encoding='gbk')
    contents = df1[contents]
    sig_dict = {}  # 后期通过sig_id 寻找smiles和signatures
    sig_ids = []
    for index, cont in contents.iterrows():
        sig_id = str(cont[0])
        sig = df2[sig_id].values.reshape(-1)
        label = cont[2:].values
        sig_dict[sig_id] = {'SMILES': cont[1], 'vectors': sig}
        sig_ids.append(sig_id)
        labels.append(label)

    return sig_dict, sig_ids, labels  # 返回一个字典{}，一个DrugID或SigID列表，一个label列表



class DataTransfer:
    def __init__(self):
        self.maxNumAtoms = 80
        self.maxNumSMI = 180
        self.maxNumPRO = 500

        self.MolTokens = ['<', '>', '#', '%', ')', '(', '+', '-', '/', '.', '[', ']', '=', '@', '\\', '\n'] +\
                      ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] +\
                      ['A', 'a', 'B', 'b', 'C', 'c', 'd', 'e', 'F', 'G', 'g', 'H', 'I', 'i', 'K', 'L', 'l',
                       'M', 'm', 'N', 'n', 'O', 'o', 'P', 'p', 'r', 'S', 's', 'T', 't', 'u', 'V', 'X', 'Z']

        self.ProTokens = ['<', 'H', 'D', 'R', 'F', 'A', 'C', 'G', 'Q', 'E', 'K', 'L', 'M', 'N', 'S', 'Y', 'T',
                          'I', 'W', 'P', 'V', 'U']

        self.descriptor = ['HeavyAtomCount', 'NumRotatableBonds', 'NumHAcceptors', 'MolWt', 'NumHDonors',
                           'NumValenceElectrons', 'TPSA', 'MolLogP', 'RingCount']

        self.m_token2index = dict((token, i) for i, token in enumerate(self.MolTokens))
        self.p_token2index = dict((token, i) for i, token in enumerate(self.ProTokens))
        self.num_m_tokens = len(self.MolTokens)
        self.num_p_tokens = len(self.ProTokens)

    def _one_of_k_encoding(self, x, allowable_set):
        if x not in allowable_set:
            raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
        # print list((map(lambda s: x == s, allowable_set)))
        return list(map(lambda s: x == s, allowable_set))

    def _one_of_k_encoding_unk(self, x, allowable_set):
        """Maps inputs not in the allowable set to the last element."""
        if x not in allowable_set:
            x = allowable_set[-1]
        return list(map(lambda s: x == s, allowable_set))

    def _atom_feature(self, atom):
        return np.array(self._one_of_k_encoding_unk(atom.GetSymbol(),
                                              ['C', 'N', 'O', 'S', 'F', 'H', 'Si', 'P', 'Cl', 'Br',
                                               'Li', 'Na', 'K', 'Mg', 'Ca', 'Fe', 'As', 'Al', 'I', 'B',
                                               'V', 'Tl', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn',
                                               'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'Mn', 'Cr', 'Pt', 'Hg', 'Pb']) +
                        self._one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
                        self._one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                        self._one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
                        [atom.GetIsAromatic()])  # (40, 6, 5, 6, 1)

    def smi2des(self, smi):
        iMol = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(iMol, radius=2, nBits=256)
        # fp = AllChem.GetMACCSKeysFingerprint(iMol)  # structure
        desc_calc = MoleculeDescriptors.MolecularDescriptorCalculator(self.descriptor)
        iDesc = desc_calc.CalcDescriptors(iMol)
        iDesc = [np.around(ides, 1) for ides in iDesc]
        iDesc = np.hstack((iDesc, fp))
        return iDesc

    def smi2graph(self, smi):
        iMol = Chem.MolFromSmiles(smi.strip())
        iAdjTmp = rdmolops.GetAdjacencyMatrix(iMol)
        # Feature
        if (iAdjTmp.shape[0] >= self.maxNumAtoms):
            iAdjTmp = iAdjTmp[:self.maxNumAtoms, :self.maxNumAtoms]
        # print('buge-1---',iAdjTmp)
        # Feature-preprocessing
        iFeature = np.zeros((self.maxNumAtoms, 58))
        iFeatureTmp = []
        k = 0
        for atom in iMol.GetAtoms():
            if k < self.maxNumAtoms:
                try:
                    iFeatureTmp.append(self._atom_feature(atom))  ### atom features only
                    k += 1
                except:
                    # print("wrong data:", i.strip(), "wrong_atom:", atom.GetSymbol())
                    k += 1

        iFeature[0:len(iFeatureTmp), 0:58] = iFeatureTmp  # 0 padding for feature-set

        # Adj-preprocessing
        iAdj = np.zeros((self.maxNumAtoms, self.maxNumAtoms))
        if len(iAdjTmp) == len(iFeatureTmp):
            iAdj[0:len(iFeatureTmp), 0:len(iFeatureTmp)] = iAdjTmp + np.eye(len(iFeatureTmp))  # 邻接矩阵+对角为1矩阵
        elif len(iAdjTmp) > len(iFeatureTmp):
            iAdj[0:len(iFeatureTmp), 0:len(iFeatureTmp)] = iAdjTmp[0:len(iFeatureTmp), 0:len(iFeatureTmp)] + np.eye(
                len(iFeatureTmp))  # 邻接矩阵+对角为1矩阵
        else:
            iAdj[0:len(iAdjTmp), 0:len(iAdjTmp)] = iAdjTmp + np.eye(len(iFeatureTmp))[0:len(iAdjTmp), 0:len(iAdjTmp)]

        return iFeature, iAdj

    def id2fps(self, ids, data_dict):
        data_fps = []
        data_vectors = []
        data_labels = []
        for i in ids:
            smi = data_dict[i]['SMILES']
            vector = data_dict[i]['vector']
            label = data_dict[i]['label']
            iMol = Chem.MolFromSmiles(smi)
            fp = AllChem.GetMACCSKeysFingerprint(iMol)  # better
            # print(len(fp))
            # fp = AllChem.RDKFingerprint(iMol)  # worse
            # fp = AllChem.GetHashedAtomPairFingerprintAsBitVect(iMol)  # worse
            # fp = AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(iMol)  # better
            # fp = AllChem.PatternFingerprint(iMol)  # worse
            # fp = AllChem.LayeredFingerprint(iMol)  # worse
            # fp = AllChem.GetMorganFingerprintAsBitVect(iMol, 2)  # worse
            data_fps.append(fp)
            data_vectors.append(vector)
            data_labels.append(label)

        fps = np.array(data_fps)
        data_vectors = np.array(data_vectors)
        data_labels = np.asarray(data_labels)
        return fps, data_vectors, data_labels

    def id2graph(self, ids, data_dict):
        data_vectors = []
        graph_features = []
        graph_adj = []

        for i in ids:
            smi = data_dict[i]['SMILES']
            vector = data_dict[i]['vector']
            iFeature, iAdj = self.smi2graph(smi)

            data_vectors.append(vector)
            graph_features.append(iFeature)
            graph_adj.append(iAdj)

        graph_features = np.asarray(graph_features)
        data_vectors = np.asarray(data_vectors)
        return graph_features, graph_adj, data_vectors

    def id2linc(self, ids, data_dict):
        MODE = 'descriptor'   # GCN, GRU
        signatures = []
        lowDimensions = []
        drug_features = []
        graph_adj = []
        data_labels = []
        for i in ids:
            smi = data_dict[i]['SMILES']
            sigs = data_dict[i]['vector']['sig']
            lowDims = data_dict[i]['vector']['lowDim']
            label = data_dict[i]['label']
            # print(vector)
            for sm, si, lo, la in zip(smi, sigs, lowDims, label):
                if MODE == 'GCN':
                    iFeature, iAdj = self.smi2graph(sm)
                    graph_adj.append(iAdj)
                if MODE == 'GRU':
                    iFeature = self.smi2token(sm)
                else:
                    iFeature = self.smi2des(sm)

                drug_features.append(iFeature)
                signatures.append(si)
                lowDimensions.append(lo)
                data_labels.append(la)
        # print(len(data_labels))
        drug_features = np.asarray(drug_features)
        signatures = np.asarray(signatures)
        lowDimensions = np.asarray(lowDimensions)
        graph_adj = np.asarray(graph_adj)
        # data_labels = np.array(data_labels).astype(float)
        return drug_features, graph_adj, signatures, lowDimensions, data_labels

    def smi2token(self, data):
        assert isinstance(data, str), 'string type of one SMILES strings required'

        # self.input_dim = self.maxNumWord

        self.pad_token = '<'
        # self.start_token = '<'
        # self.end_token = '>'

        def token2id(string, tokens):
            smi2id = np.zeros(len(string))
            for c in range(len(string)):
                smi2id[c] = tokens.index(string[c])
            return smi2id

        if len(data) <= self.maxNumSMI:
            smiTokens = data + self.pad_token * (self.maxNumSMI - len(data))
        else:
            smiTokens = data[:self.maxNumSMI]
        smiIDs = token2id(smiTokens, self.MolTokens)

        # smiIDs = []
        # for i in range(len(data)):
        #     if len(data[i]) <= self.maxNumWord:
        #         smiTokens = self.start_token + data[i] + self.end_token
        #     else:
        #         smiTokens = self.start_token + data[i][:self.maxNumWord - 2] + self.end_token
        #     smi2id = token2id(smiTokens, self.tokens)
        #     smiIDs.append(smi2id)
        return smiIDs

    def id2word(self, ids, data_dict):
        data_tokens = []
        data_vectors = []
        data_labels = []

        for i in ids:
            smi = data_dict[i]['SMILES']
            smi2token = self.smi2token(smi)
            vector = data_dict[i]['vector']
            label = data_dict[i]['label']

            data_tokens.append(smi2token)
            data_vectors.append(vector)
            data_labels.append(label)
        return data_tokens, data_vectors, data_labels

    def protein2token(self, data):
        assert isinstance(data, str), 'string type of one protein strings required'
        self.pad_token = '<'

        def token2id(string, tokens):
            smi2id = np.zeros(len(string))
            for c in range(len(string)):
                smi2id[c] = tokens.index(string[c])
            return smi2id

        if len(data) <= self.maxNumPRO:
            proteinTokens = data + self.pad_token * (self.maxNumPRO - len(data))
        else:
            proteinTokens = data[:self.maxNumPRO]
        proIDs = token2id(proteinTokens, self.ProTokens)

        return proIDs

    def id2dti(self, ids, data_dict):
        mol_mode = 'graph'  # graph, string, descriptor
        protein_sequences = []
        protein_vectors = []
        drug_iFeature = []
        drug_iAdj = []
        drug_vectors = []
        labels = []

        for i in ids:
            data = data_dict[i]

            drug = data['drug']
            protein = data['protein']
            label = data['label']
            drug_vec = drug['vector']
            drug_vectors.append(drug_vec)
            drug_smi = drug['SMILES']
            # if mol_mode == 'graph':
            iFeature, iAdj = self.smi2graph(drug_smi)
            drug_iFeature.append(iFeature)
            drug_iAdj.append(iAdj)
            #
            # elif mol_mode == 'string':
            #     smi_trans = self.smi2token(drug_smi)
            # else:
            #     smi_trans = self.smi2graph(drug_smi)


            protein_vec = protein['vector']
            protein_vectors.append(protein_vec)
            protein_seq = protein['sequence']
            protein_seq = self.protein2token(protein_seq[0])
            # print(protein_seq)
            protein_sequences.append(protein_seq)

            labels.append(label)

        return drug_iFeature, drug_iAdj, drug_vectors, protein_sequences, protein_vectors, labels



class DataLoader:
    def __init__(self, data_json, cv_json):
        def json2dict(data):
            with open(data) as f:
                data = f.read()
            drug_dict = json.loads(data)
            return drug_dict

        self.data_dict = json2dict(data_json)
        self.cv_dict = json2dict(cv_json)


    def cv_split(self, trainCollection, testCollection, drug_ids, labels, cv=1):
        X_train = []
        Y_train = []
        X_test = []
        Y_test = []
        for train in trainCollection[cv]:
            # print(train)
            X_train.append(drug_ids[train])
            Y_train.append(labels[train])
        for test in testCollection[cv]:
            X_test.append(drug_ids[test])
            Y_test.append(labels[test])
        return X_train, X_test, Y_train, Y_test

    def split(self, cross_fold=1, CV=True, test_size=0.3):
        # if CV==Ture, perform cross validation, else perform random split
        DrugIDs = []
        Labels = []
        for drug_id, infos in self.data_dict.items():
            DrugIDs.append(drug_id)
            Labels.append(infos['label'])

        trainIndex, testIndex = self.cv_dict['trainIndex'], self.cv_dict['testIndex']

        if CV:
            x_train, x_validation, y_train, y_validation = self.cv_split(trainIndex, testIndex, DrugIDs, Labels, cv=cross_fold)
        else:
            x_train, x_validation, y_train, y_validation = train_test_split(DrugIDs, Labels, test_size=test_size, random_state=34)

        return x_train, x_validation, y_train, y_validation









