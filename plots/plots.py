import matplotlib.pyplot as plt
import os

# Ensure 'plots' directory exists
plot_dir = "plots"
os.makedirs(plot_dir, exist_ok=True)

# Extract data
iterations = [
    0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 
    75000, 80000, 85000, 90000, 95000, 100000, 105000, 110000, 115000, 120000, 125000, 130000, 135000, 140000, 145000, 
    150000, 155000, 160000, 165000, 170000, 175000, 180000, 185000, 190000, 195000, 200000, 205000, 210000, 215000, 220000, 
    225000, 230000, 235000
]

# Splitting into two epochs
iterations_epoch_1 = iterations
iterations_epoch_2 = [i + iterations[-1] for i in iterations]

train_loss_epoch_1 = [10.3715, 6.2233, 6.0399, 5.8848, 5.8477, 5.7118, 5.6882, 5.6489, 5.6447, 5.5671, 5.5234, 5.5146, 
                      5.4958, 5.4613, 5.4533, 5.3718, 5.3623, 5.4083, 5.3604, 5.3057, 5.3216, 5.2939, 5.3638, 5.3181, 
                      5.2948, 5.2510, 5.2500, 5.3126, 5.2735, 5.2284, 5.2306, 5.2596, 5.1995, 5.2254, 5.2030, 5.1973, 
                      5.2383, 5.2047, 5.2515, 5.2535, 5.1962, 5.1742, 5.1465, 5.1952, 5.1893, 5.1772, 5.1294, 5.2346]

val_loss_epoch_1 = [10.3715, 5.8524, 5.6229, 5.4965, 5.4376, 5.3835, 5.3335, 5.2885, 5.2916, 5.2382, 5.2348, 5.2188, 
                    5.1827, 5.2275, 5.1433, 5.1605, 5.0905, 5.1010, 5.1192, 5.0935, 5.0938, 5.0799, 5.0944, 5.0711, 
                    5.0411, 5.0632, 5.0460, 5.0696, 5.0278, 5.0450, 4.9718, 5.0095, 4.9396, 4.9877, 4.9883, 5.0220, 
                    5.0100, 4.9785, 4.9783, 4.9558, 5.0001, 4.9643, 4.9658, 4.9326, 4.9744, 4.9749, 4.9444, 4.9876]

train_loss_epoch_2 = [5.1154, 5.1836, 5.1483, 5.2020, 5.1276, 5.0862, 5.1515, 5.1220, 5.1665, 5.1035, 5.1570, 5.1031, 
                      5.1043, 5.1330, 5.0868, 5.1387, 5.1213, 5.1309, 5.0813, 5.0750, 5.0899, 5.1174, 5.1031, 5.0334, 
                      5.0457, 5.0566, 5.0819, 5.0572, 5.1288, 5.0516, 5.0138, 5.0722, 5.0987, 5.0796, 5.0715, 5.0048, 
                      5.0752, 5.0533, 5.0411, 5.0557, 5.0550, 5.0372, 5.0606, 5.0447, 5.1008, 5.0946, 5.0695, 5.0460]

val_loss_epoch_2 = [4.9681, 4.9906, 4.9674, 4.9854, 4.9067, 4.9122, 4.9094, 4.9242, 4.9482, 4.9222, 4.8632, 4.8918, 
                    4.9085, 4.8850, 4.9029, 4.8957, 4.9284, 4.9620, 4.8955, 4.9204, 4.8875, 4.9166, 4.9088, 4.8753, 
                    4.8772, 4.9087, 4.8665, 4.8486, 4.8907, 4.8573, 4.8560, 4.8716, 4.8353, 4.8555, 4.8712, 4.8562, 
                    4.8480, 4.8554, 4.8552, 4.8651, 4.9134, 4.8762, 4.8604, 4.8433, 4.8429, 4.9125, 4.8957, 4.8563]

# Plot Training Loss for both epochs
plt.figure(figsize=(8, 5))
plt.plot(iterations_epoch_1, train_loss_epoch_1, label="Epoch 1", color='blue', linestyle='-')
plt.plot(iterations_epoch_2, train_loss_epoch_2, label="Epoch 2", color='green', linestyle='-')
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.grid()
plt.savefig(os.path.join(plot_dir, "training_loss_epochs.png"))
plt.close()

# Plot Validation Loss for both epochs
plt.figure(figsize=(8, 5))
plt.plot(iterations_epoch_1, val_loss_epoch_1, label="Epoch 1", color='red', linestyle='-')
plt.plot(iterations_epoch_2, val_loss_epoch_2, label="Epoch 2", color='purple', linestyle='-')
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Validation Loss Over Epochs")
plt.legend()
plt.grid()
plt.savefig(os.path.join(plot_dir, "validation_loss_epochs.png"))
plt.close()

# Combined Training & Validation Loss for both epochs
plt.figure(figsize=(8, 5))
plt.plot(iterations_epoch_1, train_loss_epoch_1, label="Train Loss (Epoch 1)", color='blue', linestyle='-')
plt.plot(iterations_epoch_1, val_loss_epoch_1, label="Val Loss (Epoch 1)", color='red', linestyle='-')
plt.plot(iterations_epoch_2, train_loss_epoch_2, label="Train Loss (Epoch 2)", color='green', linestyle='-')
plt.plot(iterations_epoch_2, val_loss_epoch_2, label="Val Loss (Epoch 2)", color='purple', linestyle='-')
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Train & Validation Loss Over Epochs")
plt.legend()
plt.grid()
plt.savefig(os.path.join(plot_dir, "train_val_loss_epochs.png"))
plt.close()

print(f"All plots saved in '{plot_dir}/' directory.")
