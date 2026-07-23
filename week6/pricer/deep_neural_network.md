# `deep_neural_network.py` — Price Prediction DNN

A **deep residual neural network** that predicts product prices from LLM-generated summaries, built with PyTorch.

## Architecture

### `ResidualBlock(nn.Module)`
A standard pre-activation residual block:
```
Linear → LayerNorm → ReLU → Dropout → Linear → LayerNorm → + residual → ReLU
```
Uses skip connections to enable deep training without vanishing gradients.

### `DeepNeuralNetwork(nn.Module)`
Full architecture:
1. **Input layer**: `Linear(input_size → 4096) → LayerNorm → ReLU → Dropout(0.2)`
2. **8 residual blocks** (default `num_layers=10` minus input & output)
3. **Output layer**: `Linear(4096 → 1)` — single price prediction

Total parameters: **~151 million** (with `hidden_size=4096`).

## Class: `DeepNeuralNetworkRunner`

Orchestrates training, validation, and inference.

### Setup (`setup()`)
1. **Text vectorisation**: `HashingVectorizer(n_features=5000, binary=True)` on `item.summary`
2. **Target normalisation**: `log(price + 1)` → z-score normalisation (using train stats)
3. **Model initialisation**: Creates the DNN, selects device (CUDA → MPS → CPU)
4. **Loss / Optimiser / Scheduler**:
   - `L1Loss` (MAE)
   - `AdamW` (lr=0.001, weight_decay=0.01)
   - `CosineAnnealingLR` (T_max=10)
5. **DataLoader**: Batch size 64, shuffling enabled

### Training (`train(epochs=5)`)
Per epoch:
- Forward pass with gradient clipping (`max_norm=1.0`)
- Validation pass with denormalised MAE reporting (converts predictions back to dollar scale)

### Inference (`inference(item)`)
Returns `max(0, prediction)` so prices are never negative. Denormalises the log-scaled output back to dollars.

### Persistence
- `save(path)` — saves state dict via `torch.save`
- `load(path, device)` — loads state dict with `map_location` support
