from challenge.utils.dataloader import VOCDataLoaderPerson
from challenge.prune import prune_model
from challenge.tinyyolov2 import TinyYoloV2, TinyYoloV2_pruned
from challenge.evaluate import eval
import torch


def main():
    NUM_BATCHES = 5
    BATCH_SIZE = 128

    device = torch.device('cpu')
    loader_test = VOCDataLoaderPerson(data_dir='data/', train=False, batch_size=BATCH_SIZE)

    # Load model

    state_dict = torch.load('models/person_only_baseline/model_best.pt')
    pruned = prune_model(state_dict, 0.1)

    net = TinyYoloV2_pruned(1)  # Use 1 class for person
    net.load_state_dict(pruned)

    # Evaluate model

    net.eval()
    net.to(device)

    average_precision = eval(net, loader_test, device,
                             num_batches=NUM_BATCHES, plot_roc=True)

    print(f'average precision: {average_precision}')


if __name__ == '__main__':
    main()

# Boilerplate and settings
