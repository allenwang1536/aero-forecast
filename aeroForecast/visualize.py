import matplotlib.pyplot as plt

def plot_training_loss(train_losses, model_names, epochs):
    """
    Plot the training loss for each model over the epochs.
    
    Args:
        train_losses (list): A list of training loss lists for each model.
        model_names (list): A list of strings representing the model names.
        epochs (int): The number of epochs
    """
    plt.figure(figsize=(10, 6))
    for i, loss in enumerate(train_losses):
        plt.plot(range(1, epochs + 1), loss, label=model_names[i])
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Curves')
    plt.legend()
    plt.show()

def plot_test_loss(test_losses, model_names):
    """
    Plot the test loss for each model as a bar chart.
    
    Args:
        test_losses (list): A list of test loss values for each model.
        model_names (list): A list of strings representing the model names.
    """
    plt.figure(figsize=(8, 6))
    plt.bar(model_names, test_losses)
    plt.xlabel('Model')
    plt.ylabel('Test Loss')
    plt.title('Test Loss Comparison')
    plt.show()

def visualize_results(train_losses, test_losses, model_names, epochs):
    """
    Visualize the training and test losses for all models.
    
    Args:
        train_losses (list): A list of training loss lists for each model.
        test_losses (list): A list of test loss values for each model.
        model_names (list): A list of strings representing the model names.
        epochs (int): The number of epochs.
    """
    plot_training_loss(train_losses, model_names, epochs)
    plot_test_loss(test_losses, model_names)