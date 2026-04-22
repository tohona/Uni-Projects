import argparse
import cv2
import numpy as np
import onnx
import onnxruntime as rt
#from pathlib import Path
import time
#from typing import List
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
    args = parser.parse_args()

    callback = None

    # load model
    if args.model_path.endswith(".onnx"):
        # load onnx model
        model = onnx.load(args.model_path)
        onnx.checker.check_model(model)
        # create rt session
        #providers = ['CUDAExecutionProvider','CPUExecutionProvider']
        providers = ['TensorrtExecutionProvider']
        #providers=[
        #    ("CUDAExecutionProvider", #{"cudnn_conv_algo_search": "DEFAULT"}),
       #     "CPUExecutionProvider"
        #]

        sess_options = rt.SessionOptions()
        #sess_options.enable_profiling = True
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = rt.ExecutionMode.ORT_PARALLEL
        #sess_options.optimized_model_filepath = 'optmodel.onnx'
        ort_sess = rt.InferenceSession(args.model_path, sess_options=sess_options, providers=providers)
        '''        
        # Pre-allocate CUDA-accessible input/output buffers (adjust shapes as needed)
        input_buffer = rt.OrtMemoryAllocation.from_numpy(np.zeros(ort_sess.get_inputs()[0].shape, dtype=np.float32))
        output_buffer = rt.OrtMemoryAllocation.from_numpy(np.empty(ort_sess.get_outputs()[0].shape, dtype=np.float32))

        io_binding = ort_sess.io_binding()
        io_binding.bind_input('input', input_buffer)
        io_binding.bind_output('output', output_buffer)
        '''
        
        io_binding = ort_sess.io_binding()
        X_ortvalue = rt.OrtValue.ortvalue_from_numpy(
            np.zeros(ort_sess.get_inputs()[0].shape, dtype=np.float32), 'cuda', 0)
        # OnnxRuntime will copy the data over to the CUDA device if 'input' is consumed by nodes on the CUDA device
        io_binding.bind_input(name='input', 
                              device_type=X_ortvalue.device_name(), 
                              device_id=0, 
                              element_type=np.float32, 
                              shape=X_ortvalue.shape(), 
                              buffer_ptr=X_ortvalue.data_ptr()
                              )
        io_binding.bind_output('output', 'cuda')
        
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
     
    else:
        raise Exception("Model file or path is not correct")
    
    # use webcam as source
    cam = cv2.VideoCapture(0)
    # check if camera is open
    if(cam.isOpened() == False):
        raise Exception("Can't open camera")
    
    fps_history = collections.deque(maxlen=15)  
    scale = 1 / 255.0   
    
    while(cam.isOpened()):
        ret, frame = cam.read()
        if ret == True:
            now = time.time()
            # webcam res is (480,640)
            # center crop and resize to (320,320)
            resized = cv2.resize(frame[:,80:-80], (320,320))
            resized = resized.astype('float32') * scale
            
            t0 = time.time()            
            X_ortvalue = rt.OrtValue.ortvalue_from_numpy(
                np.asarray(resized).transpose(2,0,1).reshape(1,3,320,320), 'cuda', 0)    
            #X_ortvalue.update_inplace(input)        
            ort_sess.run_with_iobinding(io_binding)
            output = io_binding.copy_outputs_to_cpu()[0]  
                        
            print(f"model: {(time.time() - t0)*1000:.2f}")
            # filter boxes based on confidence score (class_score*confidence)
            output = npFilter_boxes(output, 0.4)
            #filter boxes based on overlap
            output = npNms(output, 0.25)
            # add bounding boxes
            resized = npDisplayBoxes(resized, output)
            

            #fps = f"{int(1/(time.time() - now))}"
            fps_history.append(int(1/(time.time() - now)))
            cv2.putText(resized, f"fps={int(sum(fps_history) / len(fps_history))}", (2, 25), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (100, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('Frame',resized)
            
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            
        else:
            break
    
    # Release the video capture object
    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

