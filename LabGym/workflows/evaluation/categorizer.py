"""
LabGym.workflows.evaluation.categorizer

This module implements the categorizer evaluation / inference workflow.
"""

from __future__ import annotations

# Standard library imports
from collections import deque
import datetime
import itertools
import os
import random
from pathlib import Path

# Related third party imports
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import exposure, transform
from skimage.transform import AffineTransform
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from tensorflow import keras # pylint: disable=unused-import
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import (
    Activation,
    Add,
    AveragePooling2D,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    LSTM,
    MaxPooling2D,
    TimeDistributed,
    ZeroPadding2D,
    concatenate,
    )
from keras.models import (
    Model,
    load_model,
)
from keras.utils import (
    img_to_array,
)

# Local application imports
from LabGym.workflows.training.categorizer.train import (
    DatasetFromPath,
    DatasetFromPath_AA,
)

matplotlib.use('Agg')


# ADDEDFRM LabGym.core.categorizer (Kinda custom though)
class Categorizer():
    """
    This class implements a wrapper around the categorizer evaluation / inference workflow
    """
    # ADDEDFRM LabGym.core.categorizer (Kinda custom though)
    def load(self, model_dir: os.PathLike) -> None:
        """
        Load a trained Categorizer (weights + metadata)

        Parameters:
        - model_dir: Path to the folder that contains model_parameters.txt, and <keras-saved-model>
        """
        self.model_dir = Path(model_dir)
        self.params = self._read_parameters(self.model_dir / "model_parameters.txt")
        self.model = load_model(self.model_dir)
        self._describe() # -- optional: print verbose lines

    
    # ADDEDFRM LabGym.core.categorizer (Kinda custom though)
    def inference(self, animations, pattern_images, *, batch_size=32):
        net_type = int(self.params["network"][0])
        if net_type == 0:
            return self.model.predict(pattern_images, batch_size=batch_size)
        if net_type == 1:
            return self.model.predict(animations, batch_size=batch_size)

        return self.model.predict([animations, pattern_images], batch_size=batch_size)


    # ADDEDFRM  LabGym.core.categorizer
    def test_categorizer(self,groundtruth_path,model_path,result_path=None):
        # groundtruth_path: the folder that stores all the groundtruth behavior examples, each subfolder should be a behavior category, all categories must match those in the Categorizer
        # model_path: path to the Categorizer
        # result_path: if not None, will store the testing reports in this folder

        print('Testing the selected Categorizer...')

        animations=deque()
        pattern_images=deque()
        labels=deque()

        parameters=pd.read_csv(os.path.join(model_path,'model_parameters.txt'))

        if 'dim_conv' in list(parameters.keys()):
            dim_conv=int(parameters['dim_conv'][0])
        if 'dim_tconv' in list(parameters.keys()):
            dim_tconv=int(parameters['dim_tconv'][0])
        if 'level_conv' in list(parameters.keys()):
            level_conv=int(parameters['level_conv'][0])
        if 'dim_tconv' in list(parameters.keys()):
            level_tconv=int(parameters['level_tconv'][0])
        if 'channel' in list(parameters.keys()):
            channel=int(parameters['channel'][0])
        if 'behavior_kind' in list(parameters.keys()):
            behavior_mode=int(parameters['behavior_kind'][0])
        else:
            behavior_mode=0
        if behavior_mode==0:
            print('The behavior mode of the Categorizer: Non-interactive.')
        elif behavior_mode==1:
            print('The behavior mode of the Categorizer: Interactive basic.')
        elif behavior_mode==2:
            print('The behavior mode of the Categorizer: Interactive advanced (Social distance '+str(parameters['social_distance'][0])+').')
        else:
            print('The behavior mode of the Categorizer: Static images (non-interactive).')
        network=int(parameters['network'][0])
        if network==0:
            if behavior_mode==3:
                print('The type of the Categorizer: Pattern Recognizer (Lv '+str(level_conv)+'; Shape '+str(dim_conv)+' X '+str(dim_conv)+' X '+str(channel)+').')
            else:
                print('The type of the Categorizer: Pattern Recognizer (Lv '+str(level_conv)+'; Shape '+str(dim_conv)+' X '+str(dim_conv)+' X 3).')
        if network==1:
            print('The type of the Categorizer: Animation Analyzer (Lv '+str(level_tconv)+'; Shape '+str(dim_tconv)+' X '+str(channel)+').')
        if network==2:
            print('The type of the Categorizer: Animation Analyzer (Lv '+str(level_tconv)+'; Shape '+str(dim_tconv)+' X '+str(dim_tconv)+' X '+str(channel)+') + Pattern Recognizer (Lv '+str(level_conv)+'; Shape '+str(dim_conv)+' X '+str(dim_conv)+' X 3).')
        length=int(parameters['time_step'][0])
        print('The length of a behavior example in the Categorizer: '+str(length)+' frames.')
        if int(parameters['inner_code'][0])==0:
            print('The Categorizer includes body parts in analysis with STD = '+str(parameters['std'][0])+'.')
        else:
            print('The Categorizer does not include body parts in analysis.')
        if int(parameters['background_free'][0])==0:
            print('The Categorizer does not include background in analysis.')
        else:
            print('The Categorizer includes background in analysis.')
        if 'black_background' in parameters:
            if int(parameters['black_background'][0])==1:
                print('The background is white in the Categorizer.')
        classnames=list(parameters['classnames'])
        classnames=[str(i) for i in classnames]
        print('Behavior names in the Categorizer: '+str(classnames))
        behaviornames=[i for i in os.listdir(groundtruth_path) if os.path.isdir(os.path.join(groundtruth_path,i))]
        incorrect_behaviors=list(set(behaviornames)-set(classnames))
        incorrect_classes=list(set(classnames)-set(behaviornames))
        if len(incorrect_behaviors)>0:
            print('Mismatched behavior names in testing examples: '+str(incorrect_behaviors))
        if len(incorrect_classes)>0:
            print('Unused behavior names in the Categorizer: '+str(incorrect_classes))

        if len(incorrect_behaviors)==0 and len(incorrect_classes)==0:

            for behavior in behaviornames:

                if network!=0:
                    filenames=[i for i in os.listdir(os.path.join(groundtruth_path,behavior)) if i.endswith('.avi')]
                else:
                    filenames=[i for i in os.listdir(os.path.join(groundtruth_path,behavior)) if i.endswith('.jpg')]

                for i in filenames:

                    if network!=0:

                        path_to_animation=os.path.join(groundtruth_path,behavior,i)

                        capture=cv2.VideoCapture(path_to_animation)
                        animation=deque()
                        frames=deque(maxlen=length)

                        while True:
                            retval,frame=capture.read()
                            if frame is None:
                                break
                            frames.append(frame)

                        capture.release()

                        for frame in frames:
                            frame=np.uint8(exposure.rescale_intensity(frame,out_range=(0,255)))
                            if channel==1:
                                frame=cv2.cvtColor(np.uint8(frame),cv2.COLOR_BGR2GRAY)
                            frame=cv2.resize(frame,(dim_tconv,dim_tconv),interpolation=cv2.INTER_AREA)
                            frame=img_to_array(frame)
                            animation.append(frame)

                        animations.append(np.array(animation))

                    if network!=1:

                        path_to_pattern_image=os.path.splitext(os.path.join(groundtruth_path,behavior,i))[0]+'.jpg'
                        pattern_image=cv2.imread(path_to_pattern_image)
                        if behavior_mode==3:
                            if channel==1:
                                pattern_image=cv2.cvtColor(pattern_image,cv2.COLOR_BGR2GRAY)
                        pattern_image=cv2.resize(pattern_image,(dim_conv,dim_conv),interpolation=cv2.INTER_AREA)
                        pattern_images.append(img_to_array(pattern_image))

                    labels.append(classnames.index(behavior))

            if network!=0:
                animations=np.array(animations,dtype='float32')/255.0
            pattern_images=np.array(pattern_images,dtype='float32')/255.0

            labels=np.array(labels)

            model=load_model(model_path)

            if network==0:
                predictions=model.predict(pattern_images,batch_size=32)
            elif network==1:
                predictions=model.predict(animations,batch_size=32)
            else:
                predictions=model.predict([animations,pattern_images],batch_size=32)

            if len(classnames)==2:
                predictions=[round(i[0]) for i in predictions]
                print(classification_report(labels,predictions,target_names=classnames))
                report=classification_report(labels,predictions,target_names=classnames,output_dict=True)
            else:
                print(classification_report(labels,predictions.argmax(axis=1),target_names=classnames))
                report=classification_report(labels,predictions.argmax(axis=1),target_names=classnames,output_dict=True)

            if result_path is not None:
                pd.DataFrame(report).transpose().to_excel(os.path.join(result_path,'testing_reports.xlsx'),float_format='%.2f')

            print('Testing completed!')



__all__ = ["Categorizer"]
