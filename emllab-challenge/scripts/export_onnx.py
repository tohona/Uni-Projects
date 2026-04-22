import argparse
from pathlib import Path
from challenge.tinyyolov2 import TinyYoloV2
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str,
                        help='the path to the model file to export')
    args = parser.parse_args()

    model_path = Path(args.model_path)
    save_file_name = model_path.with_suffix('').name + '_no_bnopt'
    save_file_path = model_path.with_name(save_file_name)

    sd = torch.load(model_path)

    num_classes = sd['conv9.weight'].size(0) // len(sd['anchors']) - 5
    print(f'number of classes of the model: {num_classes}')

    model = TinyYoloV2(num_classes)
    model.load_state_dict(sd)

    dummy_input = torch.randn(1, 3, 320, 320)

    #onnx_program = torch.onnx.dynamo_export(model, dummy_input)
    #onnx_program.save(str(save_file_path.with_suffix('.onnx')))
    torch.onnx.export(
        model,
        dummy_input,
        str(save_file_path.with_suffix('.onnx')),
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
    )


if __name__ == '__main__':
    main()