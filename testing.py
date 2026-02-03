from features import Features
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from features import Features

#Preparacion de los datos para training
def test(f: Features, modelObj, model_path: str):

    #Configuracion, para mandar todo el modelo a la GPU
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # max_len es la cantidad de tokens que tiene cada secuencia con padding incluido.
    #max_len = len(f.X_trn_padded[0, :])

    #Transformar los datos del Features Engineering al formato de TORCH (Tensores)
    x_trn_padded_torch = torch.from_numpy(f.X_trn_padded)
    x_tst_padded_torch = torch.from_numpy(f.X_tst_padded)

    y_trn_torch = torch.from_numpy(f.y_trn)
    y_tst_torch = torch.from_numpy(f.y_tst)
    embedding_matrix_torch = torch.tensor(f.matriz_embedding, dtype=torch.float32)

    #Casteo a flotantes, para que despues no moleste en el criterion
    y_trn_torch = y_trn_torch.float()
    y_tst_torch = y_tst_torch.float()

    # Mover los tensores al device de TORCH
    if torch.accelerator.is_available():
        x_trn_padded_torch = x_trn_padded_torch.to(device)
        x_tst_padded_torch = x_tst_padded_torch.to(device)
        y_trn_torch = y_trn_torch.to(device)
        y_tst_torch = y_tst_torch.to(device)

    # Crear datasets y dataloaders
    test_data = TensorDataset(x_tst_padded_torch, y_tst_torch)

    test_loader = DataLoader(test_data, batch_size=128)

    model = modelObj(f, embedding_matrix_torch).to(device)
    #r'TRABAJO CREATIVO/model.model'
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    #-----------------------------------EVALUACION-----------------------------------#
    criterion = nn.BCEWithLogitsLoss()
    model.eval() # Capas como el Dropout (Útil para entrenar) no se toman en cuenta.
    total_loss = 0 #para medir el BCE

    with torch.no_grad():
        total_loss = 0.0
        TP = 0 #Verdaderos positivos
        FP = 0 #Falsos positivos
        TN = 0 #Verdaderos negativos
        FN = 0 #Falsos negativos

        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()

            # Convertimos logits a 0 o 1
            preds = (torch.sigmoid(outputs) >= 0.5).float()

            # Contadores para TP,FP,TN,FN
            TP += ((preds == 1) & (y_batch == 1)).sum().item()
            FP += ((preds == 1) & (y_batch == 0)).sum().item()
            TN += ((preds == 0) & (y_batch == 0)).sum().item()
            FN += ((preds == 0) & (y_batch == 1)).sum().item()

    #Métricas
    accuracy = (TP + TN)/(TP+TN+FP+FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = (2*recall*precision)/(recall+precision)
    print(f"BCE promedio: {total_loss / len(test_loader):.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
