from features import Features
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchmetrics import F1Score
from torch.utils.data import TensorDataset, DataLoader
from red import CNN
import numpy as np
#-----------------------------------ENTRENAMIENTO-----------------------------------#

#Features Engineering
#f = Features() 

def train(f: Features, modelObj, model_save_path: str):
    #Configuracion, para mandar todo el modelo a la GPU
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # max_len es la cantidad de tokens que tiene cada secuencia con padding incluido.
    max_len = len(f.X_trn_padded[0, :])

    #Transformar los datos del Features Engineering al formato de TORCH (Tensores)
    x_trn_padded_torch = torch.from_numpy(f.X_trn_padded)
    x_val_padded_torch = torch.from_numpy(f.X_val_padded)

    y_trn_torch = torch.from_numpy(f.y_trn)
    y_val_torch = torch.from_numpy(f.y_val)
    embedding_matrix_torch = torch.tensor(f.matriz_embedding, dtype=torch.float32)

    #Casteo a flotantes, para que despues no moleste en el criterion
    y_trn_torch = y_trn_torch.float()
    y_val_torch = y_val_torch.float()

    # Mover los tensores al device de TORCH
    if torch.accelerator.is_available():
        x_trn_padded_torch = x_trn_padded_torch.to(device)
        x_val_padded_torch = x_val_padded_torch.to(device)
        y_trn_torch = y_trn_torch.to(device)
        y_val_torch = y_val_torch.to(device)

    # Crear datasets y dataloaders
    train_data = TensorDataset(x_trn_padded_torch, y_trn_torch) #Un dataset permite guardar datos para posteriormente entrenar/testear un modelo
    val_data = TensorDataset(x_val_padded_torch, y_val_torch)

    train_loader = DataLoader(train_data, batch_size=128, shuffle=True) #un dataloader es como un contenedor de un dataset, tiene batches, iteradores, etc
    val_loader = DataLoader(val_data, batch_size=128, shuffle=True)

    model = modelObj(f, embedding_matrix_torch).to(device)
    print(model)

    criterion = nn.BCEWithLogitsLoss()
    #Probar con criterion con accuracy
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    target_acc = 0.90   # detener si llegamos al 90%
    tol = 0.001
    last_acc = 0
    max_epoch = 0
    acc_plot = np.array([])
    for epoch in range(50):
        model.train()
        max_epoch += 1
        total_loss = 0
        aciertos = 0
        total = 0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)              
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Convertimos lo que tira logits a 0 o 1 usando una sigmoide truncada
        #Calculo de aciertos en torno a datos de validacion
        for x_batch, y_batch in val_loader:
            outputs = model(x_batch)    
            preds = torch.sigmoid(outputs) >= 0.5
            aciertos += (preds.float() == y_batch).sum().item()
            total += y_batch.size(0) #cantidad de filas del batch

        acc = aciertos / total
        acc_plot = np.append(acc_plot,acc)
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")
        
        # criterio de parada por accuracy
        if acc >= target_acc:
            print("Accuracy requerida alcanzada.")
            break
        if abs(acc-last_acc)<tol:
            print("Detenido por no mejora en la Accuracy")
            break
        else:
            last_acc=acc

    # Guardamos el modelo
    # r'TRABAJO CREATIVO/model.model'
    torch.save(model.state_dict(), model_save_path)

    #Graficos : 
    x = np.linspace(1,max_epoch,len(acc_plot))
    fig, ax = plt.subplots()
    ax.plot(x, acc_plot, markeredgewidth=2)
    plt.xlabel("Epocas")
    plt.ylabel("Accuracy")
    plt.title("Evolución de la accuracy durante el entrenamiento")
    plt.grid(True)
    plt.show()
