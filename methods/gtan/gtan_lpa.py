import copy

# Función para cargar un lote de datos para entrenamiento
def load_lpa_subtensor(node_feat, work_node_feat, labels, seeds, input_nodes, device):
    batch_inputs = node_feat[input_nodes].to(device)
    batch_work_inputs = {i: work_node_feat[i][input_nodes].to(
        device) for i in work_node_feat if i not in {"Labels"}}
    batch_labels = labels[seeds].to(device)
    train_labels = copy.deepcopy(labels)
    propagate_labels = train_labels[input_nodes]
    propagate_labels[:seeds.shape[0]] = 2
    return batch_inputs, batch_work_inputs, batch_labels, propagate_labels.to(device)
