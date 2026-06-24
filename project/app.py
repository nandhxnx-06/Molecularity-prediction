from flask import Flask, render_template, request
import torch
import torch.nn as nn
import pickle
import re

app = Flask(__name__)
char2idx = pickle.load(open("vocab (1).pkl", "rb"))
mean, std = pickle.load(open("stats.pkl", "rb"))
max_len = 50
def tokenize(smiles):
    return re.findall(r'Cl|Br|.', smiles)

def encode(seq):
    encoded = [char2idx.get(c, 0) for c in seq]
    return encoded[:max_len] + [0]*(max_len - len(encoded))

class TransformerModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 16)
        encoder_layer = nn.TransformerEncoderLayer(d_model=16, nhead=2)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.mean(dim=0)
        return self.fc(x).squeeze()


model = TransformerModel(len(char2idx))
model.load_state_dict(torch.load("model (1).pth", map_location=torch.device('cpu')))
model.eval()


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    smiles = request.form["smiles"]

    tokens = tokenize(smiles)
    encoded = encode(tokens)

    x = torch.tensor([encoded], dtype=torch.long)

    with torch.no_grad():
        pred = model(x).item()

    # convert back to original scale
    pred = pred * std + mean

    return render_template("index.html",
                           prediction=round(pred, 3),
                           smiles=smiles)

if __name__ == "__main__":
    app.run(debug=True)