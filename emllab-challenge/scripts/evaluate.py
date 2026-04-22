import torch
import argparse
from challenge.tinyyolov2 import TinyYoloV2, TinyYoloV2_BNOpt, TinyYoloV2_pruned
from challenge.evaluate import eval

# A subset of VOCDataLoader just for one class (person) (0)
from challenge.utils.dataloader import VOCDataLoaderPerson



BATCH_SIZE = 128  # change this if you have VRAM size issues


def main():
    # Parse command line arguments

    parser = argparse.ArgumentParser()
    parser.add_argument('model_filename', type=str,
                        help='the path to the model file to evaluate')
    parser.add_argument('--num-batches', type=int, default=-1, help='number of batches to use for evaluation (default: all)')
    parser.add_argument('--all-classes', action='store_true',
                        help='use a model which can predict all VOC classes')
    parser.add_argument('--bnopt', action='store_true',
                        help='load the model with batch norm fusion optimization enabled')
    parser.add_argument('--pruned', action='store_true',
                        help='load a pruned model')
    parser.add_argument('--gpu', action='store_true',
                        help='train on the gpu (if possible)')
    args = parser.parse_args()

    # Load model

    device = torch.device('cuda:0' if torch.cuda.is_available() and args.gpu else 'cpu')
    num_classes = 20 if args.all_classes else 1

    net_class = TinyYoloV2
    if args.bnopt:  net_class = TinyYoloV2_BNOpt
    elif args.pruned: net_class = TinyYoloV2_pruned

    net = net_class(num_classes)

    sd = torch.load(args.model_filename)
    net.load_state_dict(sd)
    net.eval()
    net.to(device)

    # Evaluate mdodel

    loader_test = VOCDataLoaderPerson(data_dir='data/', train=False, batch_size=BATCH_SIZE)

    average_precision = eval(net, loader_test, device,
                             num_batches=args.num_batches, plot_roc=True)

    print(f'Average precision: {average_precision}')


if __name__ == '__main__':
    main()
