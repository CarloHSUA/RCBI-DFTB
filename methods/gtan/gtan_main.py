import numpy as np
import dgl
import torch
import os
from scipy.io import loadmat
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_gtan_data(dataset: str, test_size: float):
    """
    Carga el grafo, características y etiquetas dados el nombre del conjunto de datos.
    :param dataset: El nombre del conjunto de datos.
    :param test_size: El tamaño del conjunto de prueba.
    :returns: características, etiquetas, grafo, características categóricas
    """
    prefix = os.path.join(os.path.dirname(__file__), "..", "..", "data/")
    if dataset == "S-FFSD":
        cat_features = ["Target", "Location", "Type"]

        df = pd.read_csv(prefix + "S-FFSDneofull.csv")
        df = df.loc[:, ~df.columns.str.contains('Unnamed')]
        data = df[df["Labels"] <= 2]
        data = data.reset_index(drop=True)
        out = []
        alls = []
        allt = []
        pair = ["Source", "Target", "Location", "Type"]
        for column in pair:
            src, tgt = [], []
            edge_per_trans = 3
            for c_id, c_df in data.groupby(column):
                c_df = c_df.sort_values(by="Time")
                df_len = len(c_df)
                sorted_idxs = c_df.index
                src.extend([sorted_idxs[i] for i in range(df_len)
                            for j in range(edge_per_trans) if i + j < df_len])
                tgt.extend([sorted_idxs[i+j] for i in range(df_len)
                            for j in range(edge_per_trans) if i + j < df_len])
            alls.extend(src)
            allt.extend(tgt)
        alls = np.array(alls)
        allt = np.array(allt)
        g = dgl.graph((alls, allt))

        cal_list = ["Source", "Target", "Location", "Type"]
        for col in cal_list:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].apply(str).values)
        feat_data = data.drop("Labels", axis=1)
        labels = data["Labels"]
        feat_data.to_csv(prefix + "S-FFSD_feat_data.csv", index=None)
        labels.to_csv(prefix + "S-FFSD_label_data.csv", index=None)
        index = list(range(len(labels)))
        g.ndata['label'] = torch.from_numpy(
            labels.to_numpy()).to(torch.long)
        g.ndata['feat'] = torch.from_numpy(
            feat_data.to_numpy()).to(torch.float32)
        graph_path = prefix+"graph-{}.bin".format(dataset)
        dgl.data.utils.save_graphs(graph_path, [g])

        train_idx, test_idx, y_train, y_test = train_test_split(index, labels, stratify=labels, test_size=test_size/2,
                                                                random_state=2, shuffle=True)

    elif dataset == "yelp":
        cat_features = []
        data_file = loadmat(prefix + 'YelpChi.mat')
        labels = pd.DataFrame(data_file['label'].flatten())[0]
        feat_data = pd.DataFrame(data_file['features'].todense().A)
        with open(prefix + 'yelp_homo_adjlists.pickle', 'rb') as file:
            homo = pickle.load(file)
        file.close()
        index = list(range(len(labels)))
        train_idx, test_idx, y_train, y_test = train_test_split(index, labels, stratify=labels, test_size=test_size,
                                                                random_state=2, shuffle=True)
        src = []
        tgt = []
        for i in homo:
            for j in homo[i]:
                src.append(i)
                tgt.append(j)
        src = np.array(src)
        tgt = np.array(tgt)
        g = dgl.graph((src, tgt))
        g.ndata['label'] = torch.from_numpy(labels.to_numpy()).to(torch.long)
        g.ndata['feat'] = torch.from_numpy(
            feat_data.to_numpy()).to(torch.float32)
        graph_path = prefix + "graph-{}.bin".format(dataset)
        dgl.data.utils.save_graphs(graph_path, [g])

    elif dataset == "amazon":
        cat_features = []
        data_file = loadmat(prefix + 'Amazon.mat')
        labels = pd.DataFrame(data_file['label'].flatten())[0]
        feat_data = pd.DataFrame(data_file['features'].todense().A)
        with open(prefix + 'amz_homo_adjlists.pickle', 'rb') as file:
            homo = pickle.load(file)
        file.close()
        index = list(range(3305, len(labels)))
        train_idx, test_idx, y_train, y_test = train_test_split(index, labels[3305:], stratify=labels[3305:],
                                                                test_size=test_size, random_state=2, shuffle=True)
        src = []
        tgt = []
        for i in homo:
            for j in homo[i]:
                src.append(i)
                tgt.append(j)
        src = np.array(src)
        tgt = np.array(tgt)
        g = dgl.graph((src, tgt))
        g.ndata['label'] = torch.from_numpy(labels.to_numpy()).to(torch.long)
        g.ndata['feat'] = torch.from_numpy(
            feat_data.to_numpy()).to(torch.float32)
        graph_path = prefix + "graph-{}.bin".format(dataset)
        dgl.data.utils.save_graphs(graph_path, [g])

    return feat_data, labels, train_idx, test_idx, g, cat_features
