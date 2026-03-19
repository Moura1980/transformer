import numpy as np

np.random.seed(42)

#preparação do modelo
def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = (Q @ K.transpose(0, 2, 1)) / np.sqrt(d_k)
    if mask is not None:
        scores = scores + mask
    return softmax(scores) @ V


def add_norm(x, sublayer_output, epsilon=1e-6):
    out  = x + sublayer_output
    mean = np.mean(out, axis=-1, keepdims=True)
    var  = np.var(out,  axis=-1, keepdims=True)
    return (out - mean) / np.sqrt(var + epsilon)


class FFN:
    def __init__(self, d_model, d_ff):
        self.W1 = np.random.randn(d_model, d_ff)   * 0.01
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff,   d_model) * 0.01
        self.b2 = np.zeros(d_model)

    def forward(self, x):
        return np.maximum(0, x @ self.W1 + self.b1) @ self.W2 + self.b2


class MecanismoDeAtencao:
    def __init__(self, d_model):
        self.WQ = np.random.randn(d_model, d_model) * 0.01
        self.WK = np.random.randn(d_model, d_model) * 0.01
        self.WV = np.random.randn(d_model, d_model) * 0.01

    def forward(self, Q_in, K_in, V_in, mask=None):
        Q = Q_in @ self.WQ
        K = K_in @ self.WK
        V = V_in @ self.WV
        return scaled_dot_product_attention(Q, K, V, mask)


def causal_mask(seq_len):
    m = np.zeros((1, seq_len, seq_len))
    m[0][np.triu_indices(seq_len, k=1)] = -np.inf
    return m

#Encoder:
class CamadaEncoder:
    def __init__(self, d_model, d_ff):
        self.atencao = MecanismoDeAtencao(d_model)
        self.ffn     = FFN(d_model, d_ff)

    def processar(self, x):
        x = add_norm(x, self.atencao.forward(x, x, x))
        x = add_norm(x, self.ffn.forward(x))
        return x


class Encoder:
    def __init__(self, d_model, d_ff, n_camadas):
        self.camadas = [CamadaEncoder(d_model, d_ff) for _ in range(n_camadas)]

    def forward(self, x):
        for camada in self.camadas:
            x = camada.processar(x)
        return x
    

#Decoder
class CamadaDecoder:
    def __init__(self, d_model, d_ff):
        self.masked_attn = MecanismoDeAtencao(d_model)
        self.cross_attn  = MecanismoDeAtencao(d_model)
        self.ffn         = FFN(d_model, d_ff)

    def processar(self, y, Z):
        seq_len = y.shape[1]
        mask = causal_mask(seq_len)

        y = add_norm(y, self.masked_attn.forward(y, y, y, mask))
        y = add_norm(y, self.cross_attn.forward(y, Z, Z))
        y = add_norm(y, self.ffn.forward(y))
        return y


class Decoder:
    def __init__(self, d_model, d_ff, n_camadas, vocab_size):
        self.camadas = [CamadaDecoder(d_model, d_ff) for _ in range(n_camadas)]
        self.proj    = np.random.randn(d_model, vocab_size) * 0.1

    def forward(self, y, Z):
        for camada in self.camadas:
            y = camada.processar(y, Z)
        logits = y @ self.proj
        return softmax(logits)



#Modelo Completo
vocab = {
    "<PAD>":       0,
    "<START>":     1,
    "<EOS>":       2,
    "O":           3,
    "Enzo":        4,
    "é":           5,
    "programador": 6,
}
id_to_word = {v: k for k, v in vocab.items()}
VOCAB_SIZE = len(vocab)

#Hiperparâmetros (foi utilizado os mesmos hiperparâmetros do encoder da primeira atividade)
D_MODEL   = 64
D_FF      = 256
N_CAMADAS = 6
MAX_SEQ   = 20

tabela_embeddings = np.random.randn(VOCAB_SIZE, D_MODEL) * 0.01

def embed(ids):
    return tabela_embeddings[ids][np.newaxis, :, :]

encoder = Encoder(D_MODEL, D_FF, N_CAMADAS)
decoder = Decoder(D_MODEL, D_FF, N_CAMADAS, VOCAB_SIZE)


frase             = ["O", "Enzo", "é", "programador"]
encoder_input_ids = [vocab[p] for p in frase]
X = embed(encoder_input_ids)

print(f"Frase de entrada : {frase}")
print(f"IDs              : {encoder_input_ids}")
print(f"Tensor X         : {X.shape}")
print(f"\nPassando pela pilha do Encoder ({N_CAMADAS} camadas)...")

Z = encoder.forward(X)
print(f"Tensor Z (memória): {Z.shape}\n")


print("auto-regressão:\n")

decoder_ids      = [vocab["<START>"]]
palavras_geradas = ["<START>"]

for step in range(MAX_SEQ):
    Y     = embed(decoder_ids)
    probs = decoder.forward(Y, Z)

    p = probs[0, -1].astype(np.float64)
    p /= p.sum()
    next_id   = int(np.random.choice(len(p), p=p))
    next_word = id_to_word[next_id]

    decoder_ids.append(next_id)
    palavras_geradas.append(next_word)
    print(f"Passo {step + 1:2d}: '{next_word}' (id={next_id})")

    if next_word == "<EOS>":
        break

print("\nSequência gerada:")
print(" ".join(palavras_geradas))

assert Z.shape == (1, len(frase), D_MODEL)
print(f"\nSanidade Z: {Z.shape}  |  vocab_size: {VOCAB_SIZE}")