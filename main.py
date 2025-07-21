import torch
import torch.nn as nn
import json


def tokenize(text):
    return text.lower().split()

with open("source/vocab.json", "r") as f:
    vocab = json.load(f)

inv_vocab = {v: k for k, v in vocab.items()}
vocab_size = len(vocab)

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleTransformer(vocab_size).to(device)
model.load_state_dict(torch.load("source/modelo_transformer.pt", map_location=device))
model.eval()


def encode_input(text, vocab):
    return torch.tensor([vocab.get(tok, vocab["<unk>"]) for tok in tokenize(text)], dtype=torch.long)

def decode_output(tokens, inv_vocab):
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.tolist()
    return ' '.join([inv_vocab.get(tok, "<unk>") for tok in tokens if tok not in [vocab["<pad>"], vocab["<bos>"], vocab["<eos>"]]])

def generate_response(model, input_text, vocab, inv_vocab, max_len=20):
    src = encode_input(input_text, vocab).unsqueeze(0).to(device)
    tgt = torch.tensor([[vocab["<bos>"]]], dtype=torch.long).to(device)

    with torch.no_grad():
        for _ in range(max_len):
            output = model(src, tgt)
            next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
            tgt = torch.cat([tgt, next_token], dim=1)
            if next_token.item() == vocab["<eos>"]:
                break

    return decode_output(tgt.squeeze(0)[1:], inv_vocab)

print("Chatbot iniciado. Digite 'sair' para encerrar.")
while True:
    entrada = input("Você: ")
    if entrada.lower() in ["sair", "exit", "quit"]:
        print("Encerrando chatbot.")
        break
    resposta = generate_response(model, entrada, vocab, inv_vocab)
    print("Bot:", resposta)
