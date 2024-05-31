import matplotlib.pyplot as plt

train_losses_file = 'training_results/training_losses.txt'
val_losses_file = 'training_results/validation_losses.txt'

train_losses = []
val_losses = []

with open(train_losses_file, 'r') as f:
    for loss in f:
        train_losses.append(float(loss))

with open(val_losses_file, 'r') as f:
    for loss in f:
        val_losses.append(float(loss))

epochs = range(0, len(train_losses))

plt.figure(figsize=(10, 6))

plt.plot(epochs, train_losses, label='Training Loss')
plt.plot(epochs, val_losses, label='Validation Loss')

plt.title('Average Training Losses per Epoch: loss=MSE, optim=Adam, lr=1e-4, weight_decay=1e-5')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.show()
