from features import Features
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

#Features Engineering
f = Features() 

#Configuracion, para mandar todo el modelo a la GPU
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# max_len es la cantidad de tokens que tiene cada secuencia con padding incluido.
max_len = len(f.X_trn_padded[0, :])

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
train_data = TensorDataset(x_trn_padded_torch, y_trn_torch) #Un dataset permite guardar datos para posteriormente entrenar/testear un modelo
test_data = TensorDataset(x_tst_padded_torch, y_tst_torch)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True) #un dataloader es como un contenedor de un dataset, tiene batches, iteradores, etc
test_loader = DataLoader(test_data, batch_size=128)

#Clase RED CONVOLUCIONAL
class CNN(nn.Module):
    def __init__(self, embedding_matrix, num_filters=128, kernel_size=5,hidden_dim=64):
        super().__init__()
        self.num_filters=num_filters
        self.kernel_size=kernel_size
        vocab_size, embedding_dim = embedding_matrix.shape
        # El siguiente embedding se utiliza como paso del forward, para pasar de un vector de tokens a un vector denso.
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=True,       # No entrenamos embeddings
            padding_idx=0      # El 0 es el padding
        )

        # Capa Convolucional, nos entrega un mapa de caracteristicas (dim=128) que se usan como filtros
        self.Conv1=nn.Conv1d(in_channels=embedding_dim, out_channels=128, kernel_size=5,stride=1)

        # Capa de POOLING, Se reduce el mapa de caracteristicas mediante muestreo (en este caso de valor maximo).
        # Kernel: Tamaño de ventana.
        # Stride: Paso para tomar la ventana.
        # Por cada ventana toma el máximo.

        # Downsampling para velocidad.
        self.Pool1d = nn.MaxPool1d(kernel_size=2,stride=2)  

        # Capa de DROPOUT, convierto a 0 elementos randoms de los inputs con probabilidad p
        # Hinton comprobo que esto ayuda a evitar el sobreentrenamiento/overfitting.

        # Es para que la red no dependa de neuronas específicas. Apagando algunas neuronas, obligo a las otras a compensar
        self.Dropout=nn.Dropout(p=0.5)

        # Capa de Flatten, transforma todo a un tensor de 1 dimension
        self.flatten=nn.Flatten()
        
        # Capa Dense, Funcion lineal rectificadora , negativos a 0 
        self.ReLU=nn.ReLU()

        # Output sigmoide de toda la vida
        self.Sigmoid=nn.Sigmoid()

        # GRU: contexto secuencial
        self.gru = nn.GRU(
            input_size=self.num_filters,   # igual a out_channels de la CNN
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=False
        )

        # Calculo de dimensiones para poder crear la fully connected bien
        L = max_len  
        L_conv= L - 5 +1
        L_out = L_conv//2
        # Fully Connected: Del mapa de características aplanado, se conecta completamente a una capa de perceptrones con salida lineal.
        self.fc = nn.Linear(hidden_dim,1) 
    
    def forward(self, x):
        # x: (batch_size, seq_len)
        x = self.embedding(x)
        # x = self.embedding(x)  ->   (batch, seq_len) → (batch, seq_len, emb_dim)   
        # x = x.permute(0, 2, 1) ->   Reorganiza: (batch, emb_dim, seq_len)
        x = x.permute(0, 2, 1) # Se permuta simplemente para adaptar los datos de 'self.embedding(x)' al input esperado de la capa convolucional.      
        x = self.Conv1(x)
        x = self.ReLU(x)
        x = self.Pool1d(x)
        x = self.Dropout(x)
        # Adaptar para GRU
        x = x.permute(0, 2, 1)        # (B, L_out, num_filters)

        # GRU
        out, _ = self.gru(x)          # out: (B, L_out, hidden_dim)
        out = out[:, -1, :]           # tomamos la última salida temporal (contexto global)

        # Clasificación
        out = self.fc(out)            # (B, 1)
        return out

model = CNN(embedding_matrix_torch).to(device)
print(model)

# Ver video de que es BINARY CROSS ENTROPY
# BCELoss -> No usa sigmoide
# BCEWithLogitsLoss -> Tiene sigmoide incluída que es numéricamente más estable.

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # Preguntarle a di persia que prefiere
#optimizer = torch.optim.ASGD(model.parameters(), lr=1e-3) #
target_acc = 0.95   # detener si llegamos al 95%
last_acc = 0
for epoch in range(50):
    model.train()
    total_loss = 0
    correct = 0
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
        preds = torch.sigmoid(outputs) >= 0.5
        correct += (preds.float() == y_batch).sum().item()
        total += y_batch.size(0)

    acc = correct / total
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")

    # criterio de parada por accuracy
    if acc >= target_acc:
        print("Accuracy requerida alcanzada.")
        break

