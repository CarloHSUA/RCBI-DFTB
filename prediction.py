import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt 
from methods.gtan.gtan_model import GraphAttnModel
from methods.gtan.gtan_main import load_gtan_data
from methods.gtan.gtan_lpa import load_lpa_subtensor
from dgl.dataloading import MultiLayerFullNeighborSampler, DataLoader
from methods.gtan import *

PLOTS = False

# Cargamos los datos 
feat_data, labels, train_idx, test_idx, graph, cat_features = load_gtan_data('S-FFSD', 0.4)


# Configuración del dispositivo CUDA
device = 'cuda:0'
graph = graph.to(device)


# Preparación de datos y características para el modelo
num_feat = torch.from_numpy(feat_data.values).float().to(device)
cat_feat = {col: torch.from_numpy(feat_data[col].values).long().to(
    device) for col in cat_features}

# Preparación de etiquetas para el modelo
y = labels
labels = torch.from_numpy(y.values).long().to(device)


# Configuración del modelo Graph Attention
model = GraphAttnModel(in_feats=feat_data.shape[1],
                        hidden_dim = 256//4,
                        n_classes=2,
                        heads=[4]*2,  # [4,4,4]
                        activation=nn.PReLU(),
                        n_layers=2,
                        drop=[0.2, 0.1],
                        device=device,
                        gated=True,
                        ref_df=feat_data.iloc[train_idx],
                        cat_features=cat_feat).to(device)


# Configuración y carga de datos para la prueba final
test_ind = torch.from_numpy(np.array(test_idx)).long().to(device)
test_sampler = MultiLayerFullNeighborSampler(2)
test_dataloader = DataLoader(graph,
                            test_ind,
                            test_sampler,
                            use_ddp=False,
                            device=device,
                            batch_size=128,
                            shuffle=True,
                            drop_last=False,
                            num_workers=0,)


# Cargar modelo pre-entrenado
state_dict = torch.load('./new_model/new_gtan_ckpt.pth')

# Eliminar claves no deseadas
unwanted_keys = ["n2v_mlp.emb_dict.Target.weight", "n2v_mlp.emb_dict.Location.weight", "n2v_mlp.emb_dict.Type.weight"]
state_dict = {key: value for key, value in state_dict.items() if key not in unwanted_keys}
model.load_state_dict(state_dict, strict=False)

model.eval()
all_predictions = []

with torch.no_grad():
    for step, (input_nodes, seeds, blocks) in enumerate(test_dataloader):
        batch_inputs, batch_work_inputs, batch_labels, lpa_labels = load_lpa_subtensor(num_feat, cat_feat, labels,
                                                                                        seeds, input_nodes, device)

        blocks = [block.to(device) for block in blocks]
        test_batch_logits = model(
            blocks, batch_inputs, lpa_labels, batch_work_inputs)

        # Asumiendo que es una clasificación, puedes aplicar softmax para obtener probabilidades
        test_batch_probs = torch.softmax(test_batch_logits, dim=1)
        
        # Agregar las probabilidades a la lista
        all_predictions.append(test_batch_probs.cpu().numpy())

# Concatenar predicciones de todos los lotes
all_predictions = np.concatenate(all_predictions, axis=0)

# Asumiendo que es una clasificación binaria, puedes convertir las probabilidades en clases
predicted_classes = np.argmax(all_predictions, axis=1)

# Imprimir o utilizar las predicciones según sea necesario
fraud = []
no_fraud = []
for prediction in all_predictions:
    if prediction[0] > prediction[1]:
        no_fraud.append(prediction)
    else:
        fraud.append(prediction)
    print("Predicciones:", prediction, 'No fraud' if prediction[0] > prediction[1] else 'Fraud')


print("Fraudes totales ", len(fraud))
print('Transacciones sin fraude ', len(no_fraud))


if PLOTS:
    
    # Crear subgráficas (1 fila, 2 columnas)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))  # 1 fila, 2 columnas

    # Primera gráfica
    axs[0].boxplot([i[0] for i in no_fraud])
    axs[0].set_title('No fraude')

    # Segunda gráfica
    axs[1].boxplot([i[1] for i in fraud])
    axs[1].set_title('Fraude')

    plt.savefig('./img/box_fraud_and_not_fraud.png')

    # Crear subgráficas (1 fila, 2 columnas)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))  # 1 fila, 2 columnas
    bordes = np.arange(0, 1.1, 0.01)

    # Primer histograma
    axs[0].hist([i[0] for i in no_fraud], bins=bordes, edgecolor='black')
    axs[0].set_title('No fraude')
    axs[0].set_xlabel('Porcentaje de certidumbre')
    axs[0].set_ylabel('Frecuencia')

    # Segundo histograma
    axs[1].hist([i[1] for i in fraud], bins=bordes, edgecolor='black', color='orange')
    axs[1].set_title('Fraude')
    axs[1].set_xlabel('Porcentaje de certidumbre')
    axs[1].set_ylabel('Frecuencia')

    plt.savefig('./img/hist_fraud_and_not_fraud.png')

    # Ajustar diseño para evitar superposiciones
    plt.tight_layout()

    # Mostrar las subgráficas
    plt.show()

