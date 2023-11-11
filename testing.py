import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from methods.gtan.gtan_model import GraphAttnModel
from methods.gtan.gtan_main import load_gtan_data
from methods.gtan.gtan_lpa import load_lpa_subtensor
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from dgl.dataloading import MultiLayerFullNeighborSampler, DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from methods.gtan import *


# Cargamos los datos 
feat_data, labels, train_idx, test_idx, graph, cat_features = load_gtan_data('S-FFSD', 0.4)


# Configuración del dispositivo CUDA
device = 'cuda:0'
graph = graph.to(device)
oof_predictions = torch.from_numpy(
    np.zeros([len(feat_data), 2])).float().to(device)
test_predictions = torch.from_numpy(
    np.zeros([len(feat_data), 2])).float().to(device)
kfold = StratifiedKFold(
    n_splits=5, shuffle=True, random_state=2023)


# Preparación de datos y características para el modelo
y_target = labels.iloc[train_idx].values
num_feat = torch.from_numpy(feat_data.values).float().to(device)
cat_feat = {col: torch.from_numpy(feat_data[col].values).long().to(
    device) for col in cat_features}


# Imprime información sobre las características y el dispositivo
print('feat_data.shape[1]', feat_data.shape[1])
print('train_idx', train_idx[0:5])
print('feat_data.iloc[train_idx]', feat_data.iloc[train_idx])
print('cat_feat', cat_feat)
print('device', device)


# Preparación de etiquetas para el modelo
y = labels
labels = torch.from_numpy(y.values).long().to(device)


# Función de perdida para problemas de clasificación (Cross Entropy Loss)
loss_fn = nn.CrossEntropyLoss().to(device)


# Bucle principal de entrenamiento y validación
for fold, (trn_idx, val_idx) in enumerate(kfold.split(feat_data.iloc[train_idx], y_target)):
    print(f'Training fold {fold + 1}')
    trn_ind, val_ind = torch.from_numpy(np.array(train_idx)[trn_idx]).long().to(
        device), torch.from_numpy(np.array(train_idx)[val_idx]).long().to(device)

    # Configuración de muestreo y carga de datos para entrenamiento
    train_sampler = MultiLayerFullNeighborSampler(2)
    train_dataloader = DataLoader(graph,
                                trn_ind,
                                train_sampler,
                                device=device,
                                use_ddp=False,
                                batch_size=128,
                                shuffle=True,
                                drop_last=False,
                                num_workers=0)
    

    # Configuración de muestreo y carga de datos para validación
    val_sampler = MultiLayerFullNeighborSampler(2)
    val_dataloader = DataLoader(graph,
                                    val_ind,
                                    val_sampler,
                                    use_ddp=False,
                                    device=device,
                                    batch_size=128,
                                    shuffle=True,
                                    drop_last=False,
                                    num_workers=0,)
    
    
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
    

    # Configuración del optimizador y programador de tasa de aprendizaje
    lr = 0.003 * np.sqrt(128/1024)  # 0.00075
    optimizer = optim.Adam(model.parameters(), lr=lr,
                            weight_decay=1e-4)
    lr_scheduler = MultiStepLR(optimizer=optimizer, milestones=[
                                4000, 12000], gamma=0.3)
    

    # Configuración del objeto de parada temprana
    earlystoper = early_stopper(
        patience=10, verbose=True)
    start_epoch, max_epochs = 0, 2000


    # Bucle de entrenamiento por épocas
    for epoch in range(start_epoch, 15):
        train_loss_list = []
        model.train()

        # Bucle de entrenamiento por lotes
        for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
            # Carga de datos para el lote actual
            batch_inputs, batch_work_inputs, batch_labels, lpa_labels = load_lpa_subtensor(num_feat, cat_feat, labels,
                                                                                            seeds, input_nodes, device)
            blocks = [block.to(device) for block in blocks]
            train_batch_logits = model(
                blocks, batch_inputs, lpa_labels, batch_work_inputs)
            mask = batch_labels == 2
            train_batch_logits = train_batch_logits[~mask]
            batch_labels = batch_labels[~mask]

            train_loss = loss_fn(train_batch_logits, batch_labels)

            # Retropropagación y actualización de parámetros
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            lr_scheduler.step()
            train_loss_list.append(train_loss.cpu().detach().numpy())

            # Imprime información sobre el entrenamiento cada 10 lotes
            if step % 10 == 0:
                tr_batch_pred = torch.sum(torch.argmax(train_batch_logits.clone(
                ).detach(), dim=1) == batch_labels) / batch_labels.shape[0]
                score = torch.softmax(train_batch_logits.clone().detach(), dim=1)[:, 1].cpu().numpy()
                
                try:
                    print('In epoch:{:03d}|batch:{:04d}, train_loss:{:4f}, '
                            'train_ap:{:.4f}, train_acc:{:.4f}, train_auc:{:.4f}'.format(epoch, step,
                                                                                        np.mean(
                                                                                            train_loss_list),
                                                                                        average_precision_score(
                                                                                            batch_labels.cpu().numpy(), score),
                                                                                        tr_batch_pred.detach(),
                                                                                        roc_auc_score(batch_labels.cpu().numpy(), score)))
                except:
                    pass


        # mini-batch for validation
        val_loss_list = 0
        val_acc_list = 0
        val_all_list = 0


        # Establece el modelo en modo de evaluación (sin retropropagación)
        model.eval()
        with torch.no_grad():
            for step, (input_nodes, seeds, blocks) in enumerate(val_dataloader):
                batch_inputs, batch_work_inputs, batch_labels, lpa_labels = load_lpa_subtensor(num_feat, cat_feat, labels,
                                                                                                seeds, input_nodes, device)

                blocks = [block.to(device) for block in blocks]
                val_batch_logits = model(
                    blocks, batch_inputs, lpa_labels, batch_work_inputs)
                oof_predictions[seeds] = val_batch_logits
                mask = batch_labels == 2
                val_batch_logits = val_batch_logits[~mask]
                batch_labels = batch_labels[~mask]

                val_loss_list = val_loss_list + \
                    loss_fn(val_batch_logits, batch_labels)
                
                val_batch_pred = torch.sum(torch.argmax(
                    val_batch_logits, dim=1) == batch_labels) / torch.tensor(batch_labels.shape[0])
                val_acc_list = val_acc_list + val_batch_pred * \
                    torch.tensor(batch_labels.shape[0])
                val_all_list = val_all_list + batch_labels.shape[0]
                if step % 10 == 0:
                    score = torch.softmax(val_batch_logits.clone().detach(), dim=1)[
                        :, 1].cpu().numpy()
                    try:
                        print('In epoch:{:03d}|batch:{:04d}, val_loss:{:4f}, val_ap:{:.4f}, '
                                'val_acc:{:.4f}, val_auc:{:.4f}'.format(epoch,
                                                                        step,
                                                                        val_loss_list/val_all_list,
                                                                        average_precision_score(
                                                                            batch_labels.cpu().numpy(), score),
                                                                        val_batch_pred.detach(),
                                                                        roc_auc_score(batch_labels.cpu().numpy(), score)))
                    except:
                        pass


        # Valida la parada temprana y detiene el entrenamiento si es necesario
        earlystoper.earlystop(val_loss_list/val_all_list, model)
        if earlystoper.is_earlystop:
            print("Early Stopping!")
            break


    # Imprime la mejor pérdida de validación alcanzada
    print("Best val_loss is: {:.7f}".format(earlystoper.best_cv))


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
    
    

    # Configura y evalúa el mejor modelo en el conjunto de prueba
    b_model = earlystoper.best_model.to(device)

    # Guardamos el modelo en una carpeta
    torch.save(b_model.state_dict(), './new_model/new_gtan_ckpt.pth')
    
    b_model.eval()
    with torch.no_grad():
        for step, (input_nodes, seeds, blocks) in enumerate(test_dataloader):
            batch_inputs, batch_work_inputs, batch_labels, lpa_labels = load_lpa_subtensor(num_feat, cat_feat, labels,
                                                                                            seeds, input_nodes, device)

            blocks = [block.to(device) for block in blocks]
            test_batch_logits = b_model(
                blocks, batch_inputs, lpa_labels, batch_work_inputs)
            test_predictions[seeds] = test_batch_logits
            test_batch_pred = torch.sum(torch.argmax(
                test_batch_logits, dim=1) == batch_labels) / torch.tensor(batch_labels.shape[0])
            if step % 10 == 0:
                print('In test batch:{:04d}'.format(step))


# Filtra y ajusta las etiquetas del objetivo para la evaluación fuera de la muestra
mask = y_target == 2
y_target[mask] = 0


# Calcula el promedio de precisión fuera de la muestra utilizando las predicciones fuera de la muestra y las etiquetas reales
my_ap = average_precision_score(y_target, torch.softmax(
    oof_predictions, dim=1).cpu()[train_idx, 1])
print("NN out of fold AP is:", my_ap)


# Transfiere los mejores modelos y predicciones fuera de la muestra a la CPU
b_models, val_gnn_0, test_gnn_0 = earlystoper.best_model.to(
    'cpu'), oof_predictions, test_predictions


# Calcula las puntuaciones de probabilidad suavizadas y las etiquetas para el conjunto de prueba
test_score = torch.softmax(test_gnn_0, dim=1)[test_idx, 1].cpu().numpy()
y_target = labels[test_idx].cpu().numpy()
test_score1 = torch.argmax(test_gnn_0, dim=1)[test_idx].cpu().numpy()


# Filtra las etiquetas y puntuaciones para las clases distintas de la clase "2"
mask = y_target != 2
test_score = test_score[mask]
y_target = y_target[mask]
test_score1 = test_score1[mask]


# Evalúa el rendimiento en el conjunto de prueba
print("test AUC:", roc_auc_score(y_target, test_score))
print("test f1:", f1_score(y_target, test_score1, average="macro"))
print("test AP:", average_precision_score(y_target, test_score))