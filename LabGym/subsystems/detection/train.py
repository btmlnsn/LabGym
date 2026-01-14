"""
LabGym.subsystems.detection.train

This module implements the detector-training workflow.
SAFE TO IMPORT from any layer; no GUI dependencies.
"""

from __future__ import annotations

# Standard library imports.
import json
import os

# Related third party imports.
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
class DetectorTrainer():
    """
    This class implements the detector-training workflow.
    """
    # ADDEDFRM LabGym.core.detector
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' # whether the GPU is available, if so, use GPU
        self.animal_mapping = None # the animal categories and names in a Detector
        self.current_detector = None # the current Detector used for inference


    # ADDEDFRM LabGym.core.detector
    def train(self, path_to_annotation, path_to_trainingimages, path_to_detector, iteration_num, inference_size):
        
        if str('LabGym_detector_train') in DatasetCatalog.list():
            DatasetCatalog.remove('LabGym_detector_train')
            MetadataCatalog.remove('LabGym_detector_train')
            
        register_coco_instances('LabGym_detector_train',{},path_to_annotation,path_to_trainingimages)
        
        datasetcat=DatasetCatalog.get('LabGym_detector_train')
        metadatacat=MetadataCatalog.get('LabGym_detector_train')

        classnames=metadatacat.thing_classes

        model_parameters_dict={}
        model_parameters_dict['animal_names']=[]

        annotation_data=json.load(open(path_to_annotation))

        for i in annotation_data['categories']:
            if i['id']>0:
                model_parameters_dict['animal_names'].append(i['name'])

        print('Animal names in annotation file: '+str(model_parameters_dict['animal_names']))

        cfg=get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
        cfg.OUTPUT_DIR=path_to_detector
        cfg.DATASETS.TRAIN=('LabGym_detector_train',)
        cfg.DATASETS.TEST=()
        cfg.DATALOADER.NUM_WORKERS=4
        cfg.MODEL.WEIGHTS=model_zoo.get_checkpoint_url('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE=128
        cfg.MODEL.ROI_HEADS.NUM_CLASSES=int(len(classnames))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=0.5
        cfg.SOLVER.MAX_ITER=int(iteration_num)
        cfg.SOLVER.BASE_LR=0.001
        cfg.SOLVER.WARMUP_ITERS=int(iteration_num*0.1)
        cfg.SOLVER.STEPS=(int(iteration_num*0.4),int(iteration_num*0.8))
        cfg.SOLVER.GAMMA=0.5
        cfg.SOLVER.IMS_PER_BATCH=4
        cfg.MODEL.DEVICE=self.device
        cfg.SOLVER.CHECKPOINT_PERIOD=10000000000
        cfg.INPUT.MIN_SIZE_TEST=int(inference_size)
        cfg.INPUT.MAX_SIZE_TEST=int(inference_size)
        cfg.INPUT.MIN_SIZE_TRAIN=(int(inference_size),)
        cfg.INPUT.MAX_SIZE_TRAIN=int(inference_size)
        os.makedirs(cfg.OUTPUT_DIR)

        trainer=DefaultTrainer(cfg)
        trainer.resume_or_load(False)
        trainer.train()

        model_parameters=os.path.join(cfg.OUTPUT_DIR,'model_parameters.txt')

        model_parameters_dict['animal_mapping']={}
        model_parameters_dict['inferencing_framesize']=int(inference_size)

        for i in range(len(classnames)):
            model_parameters_dict['animal_mapping'][i]=classnames[i]

        with open(model_parameters,'w') as f:
            f.write(json.dumps(model_parameters_dict))

        predictor=DefaultPredictor(cfg)
        model=predictor.model

        DetectionCheckpointer(model).resume_or_load(os.path.join(cfg.OUTPUT_DIR,'model_final.pth'))
        model.eval()

        config=os.path.join(cfg.OUTPUT_DIR,'config.yaml')

        with open(config,'w') as f:
            f.write(cfg.dump())

        print('Detector training completed!')



__all__ = ['DetectorTrainer']
