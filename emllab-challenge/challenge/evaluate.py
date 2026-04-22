import torch
import tqdm

from challenge.tinyyolov2 import TinyYoloV2
from challenge.utils.ap import precision_recall_levels, ap, display_roc
from challenge.utils.yolo import nms, filter_boxes


def eval(net: TinyYoloV2, loader: torch.utils.data.DataLoader, device: torch.device,
         num_batches: int = -1, plot_roc: bool = False, leave_pbar: bool = True) -> float:
    """
    Evaluate the performance of a model.

    :param net: Model object
    :param loader: Loader object
    :param device: Device to run the model on
    :param num_batches: Number of batches
    :param plot_roc: If True, plot the ROC curve
    :return: Average precision
    """
    if (num_batches == -1):
        num_batches = len(loader)

    net.eval()

    precision = []
    recall = []
    with torch.no_grad():
        for idx, (input, target) in tqdm.tqdm(enumerate(loader), total=num_batches, leave=leave_pbar):
            if idx == num_batches:
                break

            input = input.to(device)

            output = net(input, yolo=True)

            # evaluation is done more efficiently on the cpu
            output = output.to('cpu')

            # The right threshold values can be adjusted for the target application
            output = filter_boxes(output, 0.1)
            output = nms(output, 0.5)

            for i in range(len(output)):
                p, r = precision_recall_levels(target[i], output[i])
                precision.append(p)
                recall.append(r)

    if (plot_roc):
        display_roc(precision, recall)

    return ap(precision, recall)
