import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from challenge.utils.dataloader import VOCDataLoaderPerson
from challenge.evaluate import eval
import os.path
from challenge.tinyyolov2 import TinyYoloV2_pruned

TEST_BATCH_SIZE = 128

# edit these three variables
PRUNE_DIRECTORY = 'models/person_only_both_datasets_4_layers_finetuned/iterative_pruning_15_epochs_fresh_retrain/'
PRUNE_RATE = 0.07
ORIGINAL_AP = 0.5818140640304876

def main():
    device = torch.device('cuda')
    # device = torch.device('cpu')

    loader_test = VOCDataLoaderPerson(data_dir='data/', train=False, batch_size=TEST_BATCH_SIZE)

    iterations = [0]
    average_precisions = [ORIGINAL_AP]

    model = TinyYoloV2_pruned(num_classes=1)

    for iteration in (pbar := tqdm(range(30), position=0, leave=True)):
        pbar.set_description('Pruning iterations')

        model_path = os.path.join(PRUNE_DIRECTORY, f'model_pruned_{iteration}.pt')
        if not os.path.exists(model_path):
            continue

        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        model.to(device)

        average_precisions.append(
            eval(model, loader_test, device, leave_pbar=False))
        iterations.append(iteration)
        
    
    average_precisions = np.array(average_precisions)
    iterations = np.array(iterations)

    df = pd.DataFrame(
        {'iterations': iterations, 'average_precision': average_precisions, 'prune_rate': np.zeros(len(iterations)) + PRUNE_RATE})

    # Save data
    df.to_csv(os.path.join(PRUNE_DIRECTORY, 'results_updated.csv'))

    plt.plot(df['iterations'], df['average_precision'])
    plt.xlabel('Pruning Iterations')
    plt.ylabel('Average precision')
    plt.show()

    #plt.savefig(os.path.join(PRUNE_DIRECTORY, 'results_updated.png'))
        


if __name__ == "__main__":
    main()
