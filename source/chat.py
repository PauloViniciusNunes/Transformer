import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# Base de dados
df = pd.read_csv("base.csv")

# Remove linhas com perguntas ou respostas vazias
df = df.dropna(subset=["pergunta", "resposta"])

# Garante que tudo é string
df["pergunta"] = df["pergunta"].astype(str)
df["resposta"] = df["resposta"].astype(str)

# Tokenização
def tokenize(text):
    if not isinstance(text, str):
        return []
    return text.lower().split()

def tokenize(text):
    return text.lower().split()

# Vocab
vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
idx = 4
for sent in df["pergunta"].tolist() + df["resposta"].tolist():
    for token in tokenize(sent):
        if token not in vocab:
            vocab[token] = idx
            idx += 1

inv_vocab = {v: k for k, v in vocab.items()}

# Dataset
class PizzariaDataset(Dataset):
    def __init__(self, df, vocab):
        self.data = df
        self.vocab = vocab

    def encode(self, text):
        return [self.vocab.get(tok, self.vocab["<unk>"]) for tok in tokenize(text)]

    def __getitem__(self, idx):
        src = self.encode(self.data.iloc[idx]["pergunta"])
        tgt = [self.vocab["<bos>"]] + self.encode(self.data.iloc[idx]["resposta"]) + [self.vocab["<eos>"]]
        return torch.tensor(src), torch.tensor(tgt)

    def __len__(self):
        return len(self.data)

def collate(batch):
    srcs, tgts = zip(*batch)
    srcs = nn.utils.rnn.pad_sequence(srcs, batch_first=True, padding_value=vocab["<pad>"])
    tgts = nn.utils.rnn.pad_sequence(tgts, batch_first=True, padding_value=vocab["<pad>"])
    return srcs, tgts

dataset = PizzariaDataset(df, vocab)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate)

# Construindo Transformer
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        src_emb = self.embedding(src)
        tgt_emb = self.embedding(tgt)

        src_mask = self.transformer.generate_square_subsequent_mask(src.size(1)).to(src.device)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)

        out = self.transformer(src_emb, tgt_emb, src_mask=src_mask, tgt_mask=tgt_mask)
        return self.fc(out)

# Treinando com 500 épocas
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleTransformer(len(vocab)).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(500):
    model.train()
    total_loss = 0
    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        output = model(src, tgt[:, :-1])
        output = output.reshape(-1, output.size(-1))
        target = tgt[:, 1:].reshape(-1)

        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Época {epoch+1}/500 - Loss: {total_loss:.4f}")

# Salvando trabalho
torch.save(model.state_dict(), "modelo_transformer.pt")
import json
with open("vocab.json", "w") as f:
    json.dump(vocab, f)

# Loop para conversa
model.eval()

def encode_input(text, vocab):
    return torch.tensor([vocab.get(tok, vocab["<unk>"]) for tok in tokenize(text)], dtype=torch.long)

def decode_output(tokens, inv_vocab):
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.tolist()
    return ' '.join([inv_vocab.get(tok, "<unk>") for tok in tokens if tok not in [vocab["<pad>"], vocab["<bos>"], vocab["<eos>"]]])

def generate_response(model, input_text, vocab, inv_vocab, max_len=20):
    model.eval()
    src = encode_input(input_text, vocab).unsqueeze(0).to(device)
    tgt = torch.tensor([[vocab["<bos>"]]], dtype=torch.long).to(device)
    
    for _ in range(max_len):
        out = model(src, tgt)
        next_token = out[:, -1, :].argmax(dim=-1, keepdim=True)
        tgt = torch.cat([tgt, next_token], dim=1)
        if next_token.item() == vocab["<eos>"]:
            break
    return tgt.squeeze(0)[1:]

# Loop

while True:
    entrada = input("Usuário: ")
    if entrada.lower() in ["sair", "exit", "quit"]:
        print("Encerrando...")
        break
    with torch.no_grad():
        saida = generate_response(model, entrada, vocab, inv_vocab)
        resposta = decode_output(saida, inv_vocab)
        print("Bot:", resposta)