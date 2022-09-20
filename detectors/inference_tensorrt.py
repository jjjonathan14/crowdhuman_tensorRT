from utils.tensorrt_util import *
import yaml


class ModelFileTensorRT:
    """
    A class to represent the files associated with a model.
    """
    def __init__(self, model_directory):
        self.weightsPath = os.path.join(model_directory, cfg.TensorRT.weight_file)
        self.labels = yaml.safe_load(open(f'{cfg.infer.model_directory}/{cfg.infer.labels_file}', 'rb').read())['names']


class InferenceTensorRT:
    def __init__(self, model_file, batch_infer=False, target_gpu_id=None, overlay=None):
        ctypes.CDLL(cfg.TensorRT.plugin_lib)
        cudart.cudaDeviceSynchronize()
        # load tensor RT wrapper
        print('fffff', )
        yolov5_wrapper = YoLov5TRT(model_file, f"{cfg.infer.model_directory}/{cfg.infer.labels_file}")
        self.batch_infer = batch_infer

        # model warm up
        for i in range(50):
            thread1 = warmUpThread(yolov5_wrapper)
            thread1.start()
            thread1.join()

        # create the detector class
        self.detector = infer_video(yolov5_wrapper, batch_infer=self.batch_infer)

    def infer(self, image):
        det_bboxes, det_scores, det_names, det_classids = self.detector.run(image)
        return det_bboxes, det_scores, det_names, det_classids


