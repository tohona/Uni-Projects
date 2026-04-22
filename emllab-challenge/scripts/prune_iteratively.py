from typing import Dict, Tuple
from torchvision import torch
from challenge.prune import prune_model
from challenge.evaluate import eval
from challenge.tinyyolov2 import TinyYoloV2_pruned
from challenge.utils.dataloader import VOCDataLoaderPerson
from challenge.utils.loss import YoloLoss
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import os.path
import os


#
# Parameters
#


TRAINING_BATCH_SIZE = 64
TEST_BATCH_SIZE = 128

NUM_EPOCHS = 15
NUM_ITER = 20
PRUNE_RATE = 0.07

RETRAIN_LR = 1e-4

BASE_MODEL_PATH = 'models/person_only_both_datasets_4_layers_finetuned/model_best.pt'

RETRAIN_ITERATIVE = True


#
# Helper functions
#


def _retrain_model(model, loader, device, num_epochs):
    """
    Retrain the model after pruning

    :param model: The model to retrain
    :param loader: Dataloader for the training data
    :param device: Device to train on (cuda or cpu)
    :param num_epochs: Number of epochs to train
    """
    model.train()

    optimizer = torch.optim.Adam(
        filter(lambda x: x.requires_grad, model.parameters()), lr=RETRAIN_LR)
    criterion = YoloLoss(anchors=model.anchors)

    for _ in (pbar := tqdm(range(num_epochs), position=1, leave=False)):
        pbar.set_description('Epoch')

        for _, (input, target) in (pbar2 := tqdm(enumerate(loader), total=len(loader), position=2, leave=False)):
            pbar2.set_description('Batch')

            input, target = input.to(device), target.to(device)
            optimizer.zero_grad()

            # Yolo head is implemented in the loss for training, therefore yolo=False
            output = model(input, yolo=False)
            loss, _ = criterion(output, target)
            loss.backward()
            optimizer.step()


def _prune_iteratively(state_dict, loader_test, loader_train, device, num_iter, prune_rate, num_epochs, subdir_path) -> pd.DataFrame:
    # Initialize variables
    """
    Perform a nmber of iterations of pruning and retraining

    :param state_dict: State dict of the model to prune
    :param loader_test: Dataloader for the test data
    :param loader_train: Dataloader for the training data
    :param device: Device to train on (cuda or cpu)
    :param num_iter: Number of iterations of pruning and retraining to perform
    :param prune_rate: Prune rate for each iteration
    :param num_epochs: Number of epochs to train for each iteration
    :param subdir_path: Subdirectory to save pruned models in
    :return: state_dict of the pruned model and a dataframe with the average precision for each iteration
    """

    average_precisions = []

    model = TinyYoloV2_pruned(1)
    model.load_state_dict(state_dict)
    model.to(device)

    pruned_state_dict = state_dict

    # Prunde and retrain model iteratively

    average_precisions.append(
        eval(model, loader_test, device, leave_pbar=False))

    model_path = Path(BASE_MODEL_PATH)

    for iteration in (pbar := tqdm(range(num_iter), position=0, leave=True)):
        pbar.set_description('Pruning iterations')
        pruned_state_dict = prune_model(pruned_state_dict, prune_rate)
        model.load_state_dict(pruned_state_dict)
        model.to(device)

        _retrain_model(model, loader_train, device, num_epochs)

        average_precisions.append(
            eval(model, loader_test, device, leave_pbar=False))

        print(f'AP: {average_precisions[-1]}')
        
        pruned_state_dict2 = model.state_dict()
        # iterative: use retrained weights, one-step: use pruned weights without retraining
        if RETRAIN_ITERATIVE:
            pruned_state_dict = pruned_state_dict2

        torch.save(pruned_state_dict2, os.path.join(subdir_path, f'model_pruned_{iteration + 1}.pt'))

    # Return data

    iterations = np.arange(num_iter + 1)
    average_precisions = np.array(average_precisions)
    df = pd.DataFrame(
        {'iterations': iterations, 'average_precision': average_precisions, 'prune_rate': np.zeros(num_iter + 1) + prune_rate})

    return df


#
# Main function
#


def main():
    device = torch.device('cuda')
    # device = torch.device('cpu')

    loader_train = VOCDataLoaderPerson(data_dir='data/',
        train=True, batch_size=TRAINING_BATCH_SIZE, shuffle=True)
    loader_test = VOCDataLoaderPerson(data_dir='data/', train=False, batch_size=TEST_BATCH_SIZE)

    # Iterateively prune model
    model_path = Path(BASE_MODEL_PATH)
    subdir_path = str(model_path.absolute().with_name('iterative_pruning'))
    if not os.path.exists(subdir_path):
        os.mkdir(subdir_path)

    state_dict = torch.load(BASE_MODEL_PATH)
    df = _prune_iteratively(
        state_dict, loader_test, loader_train, device, NUM_ITER, PRUNE_RATE, NUM_EPOCHS, subdir_path)

    # Save data
    df.to_csv(os.path.join(subdir_path, 'results.csv'))

    plt.plot(df['iterations'], df['average_precision'])
    plt.xlabel('Pruning Iterations')
    plt.ylabel('Average precision')
    plt.show()


if __name__ == "__main__":
    main()
