"""
LabGym.subsystems.detection.eval

This module implements the detector evaluation / inference workflow.
"""

from __future__ import annotations

# Standard library imports
import json
import os

# Related third party imports
import cv2
import torch

# Imports from Detection backend facade (Detectron2 libraries)
from LabGym.subsystems.detection.backend import(
    model_zoo, DetectionCheckpointer, get_cfg, MetadataCatalog,
    DatasetCatalog, build_detection_test_loader, register_coco_instances,
    DefaultTrainer, DefaultPredictor, COCOEvaluator, inference_on_dataset,
    build_model, Visualizer
)


# ADDEDFRM LabGym.core.detector
class Detector():
    """
    This class implements the detector evaluation / inference workflow.
    """
    # ADDEDFRM LabGym.core.detector
    def __init__(self):
        self.device= 'cuda' if torch.cuda.is_available() else 'cpu' # whether the GPU is available, if so, use GPU
        self.animal_mapping = None # the animal categories and names in a Detector
        self.current_detector = None # the current Detector used for inference


    # ADDEDFRM LabGym.core.detector
    def test(self,path_to_annotation,path_to_testingimages,path_to_detector,output_path):

        # path_to_annotation: the path to the .json file that stores the annotations in coco format
        # path_to_testingimages: the folder that stores all the ground-truth testing images
        # output_path: the folder that stores the testing images with annotations

        if str('LabGym_detector_test') in DatasetCatalog.list():
            DatasetCatalog.remove('LabGym_detector_test')
            MetadataCatalog.remove('LabGym_detector_test')

        register_coco_instances('LabGym_detector_test',{},path_to_annotation,path_to_testingimages)

        datasetcat=DatasetCatalog.get('LabGym_detector_test')
        # metadatacat=MetadataCatalog.get('LabGym_detector_test')

        animalmapping=os.path.join(path_to_detector,'model_parameters.txt')

        with open(animalmapping) as f:
            model_parameters=f.read()

        animal_names=json.loads(model_parameters)['animal_names']
        dt_infersize=int(json.loads(model_parameters)['inferencing_framesize'])

        print('The total categories of animals in this Detector: '+str(animal_names))
        print('The inferencing framesize of this Detector: '+str(dt_infersize))

        cfg=get_cfg()
        cfg.set_new_allowed(True)
        cfg.merge_from_file(os.path.join(path_to_detector,'config.yaml'))
        cfg.MODEL.WEIGHTS=os.path.join(path_to_detector,'model_final.pth')
        cfg.MODEL.DEVICE=self.device

        predictor=DefaultPredictor(cfg)

        for d in datasetcat:
            im=cv2.imread(d['file_name'])
            outputs=predictor(im)
            v=Visualizer(im[:,:,::-1],MetadataCatalog.get('LabGym_detector_test'),scale=1.2)
            out=v.draw_instance_predictions(outputs['instances'].to('cpu'))
            cv2.imwrite(os.path.join(output_path,os.path.basename(d['file_name'])),out.get_image()[:,:,::-1])

        evaluator=COCOEvaluator('LabGym_detector_test',cfg,False,output_dir=output_path)
        val_loader=build_detection_test_loader(cfg,'LabGym_detector_test')

        inference_on_dataset(predictor.model,val_loader,evaluator)

        mAP=evaluator._results['bbox']['AP']

        print(f'The mean average precision (mAP) of the Detector is: {mAP:.4f}%.')
        print('Detector testing completed!')


    # ADDEDFRM LabGym.core.detector
    def load(self,path_to_detector,animal_kinds):

        # animal_kinds: the catgories of animals / objects to be analyzed

        config=os.path.join(path_to_detector,'config.yaml')
        detector_model=os.path.join(path_to_detector,'model_final.pth')
        animalmapping=os.path.join(path_to_detector,'model_parameters.txt')
        with open(animalmapping) as f:
            model_parameters=f.read()
        self.animal_mapping=json.loads(model_parameters)['animal_mapping']
        animal_names=json.loads(model_parameters)['animal_names']
        dt_infersize=int(json.loads(model_parameters)['inferencing_framesize'])

        print('The total categories of animals / objects in this Detector: '+str(animal_names))
        print('The animals / objects of interest in this Detector: '+str(animal_kinds))
        print('The inferencing framesize of this Detector: '+str(dt_infersize))

        cfg=get_cfg()
        cfg.set_new_allowed(True)
        cfg.merge_from_file(config)
        cfg.MODEL.DEVICE=self.device
        self.current_detector=build_model(cfg)
        DetectionCheckpointer(self.current_detector).load(detector_model)
        self.current_detector.eval()


    # ADDEDFRM LabGym.core.detector
    def inference(self,inputs):

        # inputs: images that the current Detector runs on

        with torch.no_grad():
            outputs=self.current_detector(inputs)

        return outputs



__all__ = ['Detector']
