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
import collections
from helperpost import npFilter_boxes, npNms, npDisplayBoxes


now = time.time()

# needed in callback methods and I don't want to use currying with partial()
ortsess = None
model = None


def main():
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

    callback = None

    # load model
    if args.model_path.endswith(".onnx"):
        # load onnx model
        model = onnx.load(args.model_path)
        onnx.checker.check_model(model)
        # create rt session
        providers = ['CUDAExecutionProvider','CPUExecutionProvider']
        providers = ['TensorrtExecutionProvider']
        #providers = [("CUDAExecutionProvider", {"enable_cuda_graph": '1',"cudnn_conv_use_max_workspace": '1'})]
        #providers = [('TensorrtExecutionProvider', {
        #                'device_id': 0,
        #                'trt_max_workspace_size': 2147483648,
        #                'trt_fp16_enable': False,
        #            })]
        #providers = [('CUDAExecutionProvider', {"cudnn_conv_use_max_workspace": '1','cudnn_conv_algo_search': 'EXHAUSTIVE'})]#,'CPUExecutionProvider',]

        sess_options = rt.SessionOptions()
        #sess_options.enable_profiling = True
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        #sess_options.optimized_model_filepath = 'optmodel.onnx'
        
        #ort_sess = rt.InferenceSession("./models/person_only_both_datasets_4_layers_finetuned/iterative_pruning/model_pruned_8_bnopt.ort", providers=providers)
        ort_sess = rt.InferenceSession(args.model_path, sess_options=sess_options, providers=providers)
        
        # Pre-allocate CUDA-accessible input/output buffers (adjust shapes as needed)
        input_buffer = rt.OrtMemoryAllocation.from_numpy(np.zeros(ort_sess.get_inputs()[0].shape, dtype=np.float32))
        output_buffer = rt.OrtMemoryAllocation.from_numpy(np.empty(ort_sess.get_outputs()[0].shape, dtype=np.float32))

        io_binding = ort_sess.io_binding()
        io_binding.bind_input('input', input_buffer)
        io_binding.bind_output('output', output_buffer)
        
        
        # io_binding = ort_sess.io_binding()
        '''
        input_name = ort_sess.get_inputs()[0].name
        output_name = ort_sess.get_outputs()[0].name
        input_shape = ort_sess.get_inputs()[0].shape
        output_shape = ort_sess.get_outputs()[0].shape

        # Allocate GPU Memory for Input and Output
        # (Assumptions about data type and shape)
        input_buffer = rt.OrtMemoryInfo("Cuda", rt.OrtAllocatorType.OrtDeviceAllocator, 0, np.float32, input_shape)
        input_data = rt.memoryview(input_buffer.tobytes())  # View into GPU memory

        output_shape =  rt.OrtMemoryInfo("Cuda", rt.OrtAllocatorType.OrtDeviceAllocator, 0, np.float32, output_shape)
        output_buffer = rt.memoryview(output_buffer.tobytes())  # View into GPU memory

        # Create IOBinding
        io_binding = rt.IOBinding(ort_sess)
        io_binding.bind_input(input_name, input_data) 
        io_binding.bind_output(output_name, output_buffer) '''
     
        
        
        
    elif args.model_path.endswith(".pt"):
        sd = torch.load(args.model_path)
        num_classes = 20 if args.all_classes else 1
        net_class = TinyYoloV2_BNOpt if args.bnopt else TinyYoloV2
        model = net_class(num_classes)
        # load pretrained weights
        model.load_state_dict(sd)
        #put network in evaluation mode
        model.eval()

        callback = pytorchCallback
    else:
        raise Exception("Model file or path is not correct")
    
    # use webcam as source
    #cam = cv2.VideoCapture(0, cv2.CAP_DSHOW) # windows only
    cam = cv2.VideoCapture(0)

    # check if camera is open
    if(cam.isOpened() == False):
        raise Exception("Can't open camera")

    # image test
    #image = cv2.imread('../me2.png')
    
    fps_history = collections.deque(maxlen=15)  
    factor = 1 / 255.0   
    
    while(cam.isOpened()):
        ret, frame = cam.read()
        if ret == True:
            now = time.time()
            # webcam res is (480,640)
            # center crop and resize to (320,320)
            resized = cv2.resize(frame[:,80:-80], (320,320))

             
            #print(f'max: {np.max(resized)}, min: {np.min(resized)}')
            resized = resized.astype('float32') * factor
            
            t0 = time.time()
            #resized = onnxCallback(resized, ort_sess)
            
            
            input = np.asarray(resized).transpose(2,0,1).reshape(1,3,320,320)
            '''
            X_ortvalue = rt.OrtValue.ortvalue_from_numpy(input, 'cuda', 0)
            # OnnxRuntime will copy the data over to the CUDA device if 'input' is consumed by nodes on the CUDA device
            io_binding.bind_input(name='input', device_type=X_ortvalue.device_name(), device_id=0, element_type=np.float32, shape=X_ortvalue.shape(), buffer_ptr=X_ortvalue.data_ptr())
            #io_binding.bind_cpu_input('input', input)
            io_binding.bind_output('output', 'cuda')
            ort_sess.run_with_iobinding(io_binding)
            output = io_binding.copy_outputs_to_cpu()[0]  
            '''
            # Copy preprocessed data directly to the input_data buffer
            input.tobytes(input_data) 
            ort_sess.run_with_iobinding(io_binding)
            output = np.array(output_buffer).reshape(output_shape)
            
            print(f"model: {(time.time() - t0)*1000:.2f}")
            # filter boxes based on confidence score (class_score*confidence)
            output = onnxFilter_boxes(output, 0.4)
            #filter boxes based on overlap
            output = onnxNms(output, 0.25)
            # add bounding boxes
            resized = onnxDisplayBoxes(resized, output)
            

            #fps = f"{int(1/(time.time() - now))}"
            fps_history.append(int(1/(time.time() - now)))
            cv2.putText(resized, f"fps={int(sum(fps_history) / len(fps_history))}", (2, 25), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (100, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('Frame',resized)
            
            #print(f"total: \t {(time.time() - now)*1000:.2f} ms")
            #print("------------------------")

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            
        else:
            break
    
    # Release the video capture object
    cam.release()
    cv2.destroyAllWindows()


"""
Display boxes of the detected object.

Args: 
    frame: current frame or image to draw in
    output: list of bounding boxes

Returns: frame
"""
def displayBoxes(frame, output: List[np.ndarray]):
    # frame shape is (320,320,3)
    pad = 20
    frame = np.pad(frame, ((pad, pad), (pad, pad), (0, 0)), mode='constant')
    img_shape = 320
        
    if output:
        bboxes = torch.stack(output, dim=0)
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


"""
Camera loop pipeline: forward model, bbox filtering, NMS, display

Args: 
    frame: current frame or image to draw in
    model: pytorch model

Returns: frame
"""
def pytorchCallback(frame):
    # track fps
    global now
    global model
    fps = f"{int(1/(time.time() - now))}"
    now = time.time()
        
    # convert frame to suitable model input and forward to model
    input = torch.tensor(frame, dtype=torch.float32).permute(2,0,1).view(1,3,320,320)
    output = model(input)
    
    # filter boxes based on confidence score (class_score*confidence)
    output = filter_boxes(output, 0.1)
    
    #filter boxes based on overlap
    output = nms(output, 0.25)
    
    # add bounding boxes
    frame = displayBoxes(frame, output)
    
    # display furrent fps
    cv2.putText(frame, "fps="+fps, (2, 25), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (100, 255, 0), 2, cv2.LINE_AA)
    
    return frame


"""
Camera loop pipeline: forward model, bbox filtering, NMS, display

Args: 
    frame: current frame or image to draw in
    ortsess: onnx runtime session

Returns: frame
"""
def onnxCallback(frame, ortsess):    
    input = np.asarray(frame).transpose(2,0,1).reshape(1,3,320,320)#.astype('float32')
    test = time.time()
    
    output = ortsess.run(['output'], {'input': input})
    print(f"model: \t {(time.time() - test)*1000:.2f} ms")
    
    # filter boxes based on confidence score (class_score*confidence)
    output = onnxFilter_boxes(output[0], 0.5)
    #filter boxes based on overlap
    output = onnxNms(output, 0.25)
    # add bounding boxes
    frame = onnxDisplayBoxes(frame, output)
    
    return frame


'''
TODO: change torch to numpy
'''
def onnxIou(bboxes1, bboxes2):
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
def onnxNms(filtered_ndarray: List[np.ndarray], threshold: float) -> List[np.ndarray]:
    result = []
    for x in filtered_ndarray:
        # Sort coordinates by descending confidence
        order = np.argsort(x[:, 4], 0)
        order = order[::-1]
        x = x[order]
        ious = onnxIou(x,x) # get ious between each bbox in x

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
def onnxFilter_boxes(output_ndarray: np.ndarray, threshold) -> List[np.ndarray]:
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
def onnxDisplayBoxes(frame, output: List[torch.tensor]):
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


if __name__ == '__main__':
    main()
