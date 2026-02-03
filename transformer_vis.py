
# Ejemplo de codigo para trabajar un texto en particular
# pip install seaborn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

model = AutoModelForSequenceClassification.from_pretrained("tukx/fake-news-classificator", output_attentions=True)
text = "Trump STUPIDLY Attacks A Major U.S. Ally Before Threatening North Korea With War"

tokenizer = AutoTokenizer.from_pretrained("tukx/fake-news-classificator")
inputs = tokenizer(text, return_tensors="pt")

print("Inputs: ")
print(inputs)
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
print('Tokens: ')
print(tokens)



print()
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

    attentions = outputs.attentions # Tupla de tensores de atención

#print('Atencion: ')
#print(attentions)
# Que relación tiene cada token con cada token?
# El primero 0 -> Primer batch
# Segundo 0 -> Primer head
# Tercer 0 -> Capa?

# attentions[0] = primera capa (12 capas en modelos como BERT)
# attentions[0][0] = primer batch (normalmente batch_size=1)
# attentions[0][0][0] = primer head de atención de la primera capa

# Información sobre la estructura
print(f"Número de capas: {len(attentions)}")
print(f"Forma de cada capa: {attentions[0].shape}")  # (batch_size, num_heads, seq_len, seq_len)
print(f"Número de heads: {attentions[0].shape[1]}")

# attentions[layer][batch][head] = attention_matrix

capa = 0  # Primera capa del transformer
batch = 0 # Primer (y único) ejemplo del batch
head = 0  # Primera cabeza de atención

attention_map = attentions[capa][batch][head]

print('Attention map primer head: ')
print(attention_map)
print('Attention map heads promediados: ')

attention_map_avg = attentions[0][0].mean(dim=0)

print(attention_map_avg)

plt.figure(figsize=(10, 8))
ax = sns.heatmap(attention_map_avg.squeeze().detach().numpy(), cmap="viridis")
plt.xlabel("Key Tokens")
plt.ylabel("Query Tokens")
plt.title("Attention Map")

n = len(tokens)
offset = 0.5  # medio paso

ax.set_xticks(np.arange(n) + offset)
ax.set_yticks(np.arange(n) + offset)

ax.set_xticklabels(tokens, rotation=45, ha="right")
ax.set_yticklabels(tokens, rotation=0)

plt.tight_layout()
plt.show()

predicted_class_id = logits.argmax().item()
model.config.id2label[predicted_class_id]
print('Logits: ')
print(logits)
print('Predicted Class Id: ')
print(predicted_class_id)
print('Predicted Class: ')
print(model.config.id2label[predicted_class_id])

#import numpy as np

# Mejor: promediar múltiples cabezas para mejor visualización
#def plot_attention(attentions, tokens, layer=0, batch=0):
#    # Promedio de todas las cabezas en una capa
#    attention_heads = attentions[layer][batch]  # [num_heads, seq_len, seq_len]
#    attention_avg = attention_heads.mean(dim=0)  # [seq_len, seq_len]
#    
#    plt.figure(figsize=(12, 10))
#    
#    # Crear máscara para la diagonal (opcional)
#    mask = np.eye(len(tokens)) == 0
#    
#    ax = sns.heatmap(
#        attention_avg.detach().numpy(),
#        cmap="Blues",
#        xticklabels=tokens,
#        yticklabels=tokens,
#        # mask=mask,  # Descomenta para ocultar auto-atención
#        square=True,
#        linewidths=0.5
#    )
#    
#    plt.xlabel("Key Tokens")
#    plt.ylabel("Query Tokens")
#    plt.title(f"Attention Map - Capa {layer} (Promedio de todas las cabezas)")
#    plt.xticks(rotation=45, ha='right')
#    plt.yticks(rotation=0)
#    plt.tight_layout()
#    plt.show()

# Usar la función
#plot_attention(attention_map, tokens, layer=0)