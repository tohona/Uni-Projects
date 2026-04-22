from evaluate import eval
import tqdm
from challenge.utils.loss import YoloLoss
from challenge.tinyyolov2 import TinyYoloV2
import torch
from pathlib import Path
import argparse

# A subset of VOCDataLoader just for one class (person) (0)
from challenge.utils.dataloader import VOCDataLoaderPerson, VOC_plus_COCO_DataLoaderPerson

TRAINING_BATCH_SIZE = 64

TEST_BATCH_SIZE = 128
NUM_TEST_BATCHES = 3

PATH_BASE = './models/person_only_test/'

LEARNING_RATE = 1e-3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train (default: 5)')
    parser.add_argument('--gpu', action='store_true',
                        help='train on the gpu (if possible)')
    parser.add_argument('--more-augmentation', action='store_true',
                        help='enable more augmentation techniques during training')
    parser.add_argument('--both-datasets', action='store_true',
                        help='use person images from both the VOC and COCO datasets for more training data')
    parser.add_argument('--finetune-more-layers', action='store_true',
                        help='finetune the last 4 layers instead of only the last')
    args = parser.parse_args()

    Path(f"{PATH_BASE}").mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() and args.gpu else 'cpu')
    # device = torch.device('cpu')

    print(f'Using device: {device}')
    print(f'Training configuration: {"more_augmentation " if args.more_augmentation else ""}{"both_datasets " if args.both_datasets else ""}{"finetune_more_layers " if args.finetune_more_layers else ""}')

    if args.both_datasets:
        loader = VOC_plus_COCO_DataLoaderPerson(data_dir='data/',
            train=True, batch_size=TRAINING_BATCH_SIZE, shuffle=True, more_augmentation=args.more_augmentation)
        loader_test = VOC_plus_COCO_DataLoaderPerson(data_dir='data/', train=False, batch_size=TEST_BATCH_SIZE)
    else:
        loader = VOCDataLoaderPerson(data_dir='data/',
            train=True, batch_size=TRAINING_BATCH_SIZE, shuffle=True, more_augmentation=args.more_augmentation)
        loader_test = VOCDataLoaderPerson(data_dir='data/', train=False, batch_size=TEST_BATCH_SIZE)

    #_, _ = next(iter(loader))
    #return

    # We define a tinyyolo network with only two possible classes
    net = TinyYoloV2(num_classes=1)
    sd = torch.load("./models/voc_pretrained.pt")

    # We load all parameters from the pretrained dict except for the last layer
    net.load_state_dict({k: v for k, v in sd.items()
                        if not '9' in k}, strict=False)

    if not args.finetune_more_layers:
        # net can be in eval mode because the batchnorm layers come after
        # the frozen convolution layers
        net.eval()

    net.to(device)

    # Definition of the loss
    criterion = YoloLoss(anchors=net.anchors)

    # We only train the last layer (conv9) or the last 4 layers
    freeze_layers = ['1', '2', '3', '4', '5']
    if not args.finetune_more_layers:
        freeze_layers.extend(['6', '7', '8'])
    for key, param in net.named_parameters():
        if any(x in key for x in freeze_layers):
            param.requires_grad = False

    optimizer = torch.optim.Adam(
        filter(lambda x: x.requires_grad, net.parameters()), lr=LEARNING_RATE)

    test_AP = []

    for epoch in range(args.epochs):
        if args.finetune_more_layers:
            # eval() activates net.eval(), but we want to train batchnorm layers
            net.train() 

        for idx, (input, target) in tqdm.tqdm(enumerate(loader), total=len(loader)):
            input, target = input.to(device), target.to(device)

            optimizer.zero_grad()

            # Yolo head is implemented in the loss for training, therefore yolo=False
            output = net(input, yolo=False)
            loss, _ = criterion(output, target)
            loss.backward()
            optimizer.step()

        # Calculation of average precision with collected samples
        average_precision = eval(
            net, loader_test, device, num_batches=NUM_TEST_BATCHES, plot_roc=False)
        test_AP.append(average_precision)
        print(f'[EPOCH {epoch}] average precision', average_precision)

        # save model
        torch.save(net.state_dict(), f'{PATH_BASE}/model_epoch_{epoch}.pt')

    with open(f'{PATH_BASE}/log.txt', mode='w') as log:
        log.write(
            '\n'.join(f'[EPOCH {i}]: {test_AP[i]}' for i in range(len(test_AP))))


if __name__ == "__main__":
    main()
