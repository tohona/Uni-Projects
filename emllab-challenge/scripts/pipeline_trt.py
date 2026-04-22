import argparse
import cv2
import numpy as np
import onnx
import onnxruntime as rt
from pathlib import Path
import time
from challenge.tinyyolov2 import TinyYoloV2, TinyYoloV2_BNOpt
import torch
from typing import List
from challenge.utils.viz import num_to_class
from challenge.utils.yolo import nms, filter_boxes
#import torch_tensorrt
import helpertrt
import tensorrt as trt
import sys, os
from cuda import cuda, cudart

TRT_LOGGER = trt.Logger()


now = time.time()

# needed in callback methods and I don't want to use currying with partial()
ortsess = None
model = None


def main():

    """    
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str,
                        help='The path of model file')
    parser.add_argument('--all-classes', action='store_true',
                        help='use a model which can predict all VOC classes (pytorch only)')
    parser.add_argument('--bnopt', action='store_true',
                        help='load the model with batch norm fusion optimization enabled (pytorch only)')
    #parser.add_argument('--onnx', action='store_true',
    #                    help='use onnx if used, else pytorch')
    args = parser.parse_args()
    """
    callback = None

    # load model
    """
    sd = torch.load(args.model_path)
    num_classes = 20 if args.all_classes else 1
    net_class = TinyYoloV2_BNOpt if args.bnopt else TinyYoloV2
    model = net_class(num_classes)
    # load pretrained weights
    model.load_state_dict(sd)
    #put network in evaluation mode
    model.eval()
    
    traced_model = torch.jit.trace(model, [torch.randn((1, 3, 480, 480)).to("cuda")])
    trt_model = torch_tensorrt.compile(
        traced_model,
        inputs = [torch_tensorrt.Input((1, 3, 224, 224), dtype=torch.float32)],
        enabled_precisions = {torch.float32}
    )

    """ 
    # use webcam as source
    #cam = cv2.VideoCapture(0, cv2.CAP_DSHOW) # windows only
    cam = cv2.VideoCapture(0)

    # check if camera is open
    if(cam.isOpened() == False):
        raise Exception("Can't open camera")

    # image test
    #image = cv2.imread('../me2.png')
    t0 = time.time()
    onnx_path = 'models/person_only_both_datasets_4_layers_finetuned/model_best_bnopt.onnx'#'models/person_only_baseline/model_best_bnopt.onnx'#'model_best_bnopt-sim.onnx'#
    trt_path = 'models/person_only_both_datasets_4_layers_finetuned/test2.trt' # 'modelenginev2.trt'
    engine = get_engine(onnx_path, trt_path)
    print(f"load engine: \t {(time.time() - t0)*1000:.2f} ms") 
    t0 = time.time()
    context = engine.create_execution_context()
    print(f"create context: \t {(time.time() - t0)*1000:.2f} ms") 
    t0 = time.time()
    inputs, outputs, bindings, stream = helpertrt.allocate_buffers(engine)
    print(f"alloc buf: \t {(time.time() - t0)*1000:.2f} ms") 
    
    
    while(cam.isOpened()):
        t0 = time.time()
        ret, frame = cam.read()
        print(f"cam read: {(time.time() - t0)*1000:.2f} ms")
        
        if ret == True:
            #t0 = time.time()
            # webcam res is (480,640)
            # center crop and resize to (320,320)
            resized = cv2.resize(frame[:,80:-80], (320,320))
            resized = resized.astype('float32') / 255.0
            
            resized = trtCallback(resized, context, inputs, outputs, bindings, stream)
            
            fps = f"{int(1/(time.time() - t0))}"
            cv2.putText(resized, "fps="+fps, (2, 25), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (100, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('Frame',resized)
            
            print(f"total: \t {(time.time() - t0)*1000:.2f} ms \n ------------------------")
            #print("------------------------")

            key = cv2.waitKey(20)
            if key == ord('q'):
                break
            
        else:
            break
    
    # Release the video capture object
    cam.release()
    cv2.destroyAllWindows()



'''
TODO: change torch to numpy
'''
def npIou(bboxes1, bboxes2):
    """ calculate iou between each bbox in `bboxes1` with each bbox in `bboxes2`"""
    #print(bboxes1[...,:4].reshape(-1, 4))
    #print(bboxes1[...,:4].reshape(-1, 4).shape)
    px, py, pw, ph = np.array_split(bboxes1[...,:4].reshape(-1, 4), 4, axis=-1)
    #print(bboxes2[...,:4].reshape(-1, 4))
    lx, ly, lw, lh = np.array_split(bboxes2[...,:4].reshape(-1, 4), 4, axis=-1)
    #print(px)
    #print(px.shape)
    #print(lx, ly, lw, lh)
    px1, py1, px2, py2 = px - 0.5 * pw, py - 0.5 * ph, px + 0.5 * pw, py + 0.5 * ph
    lx1, ly1, lx2, ly2 = lx - 0.5 * lw, ly - 0.5 * lh, lx + 0.5 * lw, ly + 0.5 * lh
    zero = np.array(0.0, dtype=px1.dtype)
    
    dx = np.maximum(np.minimum(px2, lx2.T) - np.maximum(px1, lx1.T), zero)
    dy = np.maximum(np.minimum(py2, ly2.T) - np.maximum(py1, ly1.T), zero)
    intersections = dx * dy
    pa = (px2 - px1) * (py2 - py1) # area
    la = (lx2 - lx1) * (ly2 - ly1) # area
    unions = (pa + la.T) - intersections
    ious = (intersections/unions).reshape(*bboxes1.shape[:-1], *bboxes2.shape[:-1])

    return ious


'''
TODO: change torch to numpy
'''
def npNms(filtered_ndarray: List[np.ndarray], threshold: float) -> List[np.ndarray]:
    result = []
    for x in filtered_ndarray:
        # Sort coordinates by descending confidence
        order = np.argsort(x[:, 4], 0)
        order = order[::-1]
        x = x[order]
        ious = npIou(x,x) # get ious between each bbox in x

        # Filter based on iou
        keep = (ious > threshold).astype(np.int_)
        keep = np.triu(keep, 1)
        keep = np.sum(keep, 0, keepdims=True)
        keep = keep.T
        keep = np.broadcast_to(keep, x.shape) == 0

        result.append(x[keep].reshape(-1, 6))
    return result


'''
TODO: change torch to numpy
'''
def npFilter_boxes(output_ndarray: np.ndarray, threshold) -> List[np.ndarray]:
    b, a, h, w, c = output_ndarray.shape
    x = output_ndarray.reshape(b, a * h * w, c)
    #print(x.shape)
    boxes = x[:, :, 0:4]
    confidence = x[:, :, 4]
    #print(x[:, :, 5:].shape)
    scores = np.max(x[:, :, 5:], axis=-1)
    idx = np.argmax(x[:, :, 5:], axis=-1)
    
    idx = np.float32(idx)
    scores = scores * confidence
    mask = scores > threshold
    
    filtered = []
    for c, s, i, m in zip(boxes, scores, idx, mask):
        if m.any():
            detected = np.concatenate((c[m, :], s[m, None], i[m, None]), -1)
        else:
            detected = np.zeros((0, 6), dtype=x.dtype)
        filtered.append(detected)
    return filtered

"""
Display boxes of the detected object.

Args: 
    frame: current frame or image to draw in
    output: list of bounding boxes

Returns: frame
"""
def npDisplayBoxes(frame, output: List[torch.tensor]):
    # frame shape is (320,320,3)
    pad = 20
    frame = np.pad(frame, ((pad, pad), (pad, pad), (0, 0)), mode='constant')
    img_shape = 320
        
    if output:
        bboxes = np.stack(output, axis=0)
        for i in range(bboxes.shape[1]):
            if bboxes[0,i,-1] >= 0:
                '''                
                Index meanings:
                0 - x center
                1 - y center
                2 - Width
                3 - Height
                4 - confidence
                5 - class idx
                '''                
                # top left corner
                start_point = (
                    int(bboxes[0,i,0]*img_shape - bboxes[0,i,2]*img_shape/2) - pad,
                    int(bboxes[0,i,1]*img_shape - bboxes[0,i,3]*img_shape/2) - pad
                )
                # bottom right corner
                end_point = (
                    int(bboxes[0,i,0]*img_shape + bboxes[0,i,2]*img_shape/2) + pad,
                    int(bboxes[0,i,1]*img_shape + bboxes[0,i,3]*img_shape/2) + pad
                )
                color = (255, 0, 0)  # BGR
                #print(num_to_class(int(bboxes[0,i,5])))
                #print((start_point, end_point))
                
                cv2.rectangle(frame, start_point, end_point, color, 2) 

    return frame


def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            helpertrt.EXPLICIT_BATCH
        ) as network, builder.create_builder_config() as config, trt.OnnxParser(
            network, TRT_LOGGER
        ) as parser, trt.Runtime(
            TRT_LOGGER
        ) as runtime:
            config.max_workspace_size = 1 << 30  # 28 = 256MiB  30 = 1GiB
            builder.max_batch_size = 1
            #builder.fp16_mode = True
            if builder.platform_has_fast_fp16:
                print("has fp16 precision")
                config.set_flag(trt.BuilderFlag.FP16)
            
            #if builder.platform_has_fast_int8:
            #    print("has int8 precision")
            #    config.set_flag(trt.BuilderFlag.INT8)
            
            print(".....")
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print(
                    "ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.".format(onnx_file_path)
                )
                exit(0)
            print("Loading ONNX file from path {}...".format(onnx_file_path))
            with open(onnx_file_path, "rb") as model:
                print("Beginning ONNX file parsing")
                if not parser.parse(model.read()):
                    print("ERROR: Failed to parse the ONNX file.")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
            # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
            network.get_input(0).shape = [1, 3, 320, 320]
            print("Completed parsing of ONNX file")
            print("Building an engine from file {}; this may take a while...".format(onnx_file_path))
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(plan)
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()


def trtCallback(frame, context, inputs, outputs, bindings, stream):
    #input = np.array(frame).transpose(2,0,1).reshape(1,3,320,320).astype('float16')
    #test = time.time()    
    inputs[0].host = np.array(frame).transpose(2,0,1).reshape(1,3,320,320)#.astype('float32')
    #np.copyto(inputs[0].host, input)
    #print(f"input to gpu: \t {(time.time() - test)*1000:.2f} ms") 
    #test = time.time()
    trt_output = helpertrt.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    #print(f"inference: \t {(time.time() - test)*1000:.2f} ms") 
        
    # trt output is a list with 3000 values
    # expected output shape is (1, 1, 5, 10, 10, 6)
    output = np.array(trt_output).reshape((1, 1, 5, 10, 10, 6))     
    # filter boxes based on confidence score (class_score*confidence)
    output = npFilter_boxes(output[0], 0.25)
    #filter boxes based on overlap
    output = npNms(output, 0.25)
    # add bounding boxes
    frame = npDisplayBoxes(frame, output)
    return frame


if __name__ == '__main__':
    main()
