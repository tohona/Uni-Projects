import torch
import argparse
from pathlib import Path
from challenge.tinyyolov2 import TinyYoloV2_BNOpt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str,
                        help='the path to the model file to evaluate')
    parser.add_argument('--export-onnx', action='store_true',
                        help='export the optimized model for onnx')
    args = parser.parse_args()

    model_path = Path(args.model_path)
    save_file_name = model_path.with_suffix('').name + '_bnopt'
    save_file_path = model_path.with_name(save_file_name)

    sd = torch.load(model_path)

    print('input:\n\n', sd.keys())

    save_sd = {
        'anchors': sd['anchors'],
        'conv9.weight': sd['conv9.weight'],
        'conv9.bias': sd['conv9.bias'],
    }

    #device = sd['conv1.weight'].device
    device = torch.device('cpu')

    for i in range(1, 9):
        conv_w = sd[f'conv{i}.weight'].to(device)
        conv_b = torch.zeros(conv_w.size(0)).to(device) # no bias
        bn_w = sd[f'bn{i}.weight'].to(device)
        bn_b = sd[f'bn{i}.bias'].to(device)
        bn_rm = sd[f'bn{i}.running_mean'].to(device)
        bn_rv = sd[f'bn{i}.running_var'].to(device)
        w, b = fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_w, bn_b)
        save_sd[f'conv{i}.weight'] = w
        save_sd[f'conv{i}.bias'] = b
    
    print('\noutput:\n\n', save_sd.keys(), '\n')

    if args.export_onnx:
        num_classes = sd['conv9.weight'].size(0) // len(sd['anchors']) - 5
        print(f'number of classes of the model: {num_classes}')

        model = TinyYoloV2_BNOpt(num_classes)
        model.load_state_dict(save_sd)

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
    else:
        torch.save(save_sd, save_file_path.with_suffix('.pt'))




def fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_w, bn_b):
    """
    Input:
        conv_w: shape=(output_channels, in_channels, kernel_size, kernel_size)
        conv_b: shape=(output_channels)
        bn_rm:  shape=(output_channels)
        bn_rv:  shape=(output_channels)
        bn_w:   shape=(output_channels)
        bn_b:   shape=(output_channels)
    
    Output:
        fused_conv_w = shape=conv_w
        fused_conv_b = shape=conv_b
    """
    bn_eps = 1e-05
    
    coef = bn_w / torch.sqrt(bn_rv + bn_eps)
    fused_weight = conv_w * coef.view((-1, 1, 1, 1))
    fused_bias = coef * (conv_b - bn_rm) + bn_b

    return fused_weight, fused_bias



if __name__ == '__main__':
    main()
