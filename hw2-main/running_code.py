
import sys
sys.path.append('./python')
sys.path.append('./apps')
sys.path.append('./needle')
import needle as ndl
from mlp_resnet import train_mnist

train_mnist(batch_size=100, epochs=2, optimizer=ndl.optim.Adam, lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="./data",)