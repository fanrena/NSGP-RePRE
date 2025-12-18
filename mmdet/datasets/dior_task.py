# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import DATASETS
from .xml_style_task import XMLTask


@DATASETS.register_module()
class DIORTask(XMLTask):
    """Dataset for PASCAL VOC."""

    METAINFO = {
        'classes':
        ('airplane', 'airport', 'baseballfield', 'basketballcourt',
        'bridge', 'chimney', 'dam', 'Expressway-Service-area', 'Expressway-toll-station',
        'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship', 'stadium', 'storagetank',
        'tenniscourt', 'trainstation', 'vehicle', 'windmill'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
                    (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
                    (153, 69, 1), (120, 166, 157), (0, 182, 199),
                    (0, 226, 252), (182, 182, 255), (0, 0, 230), (220, 20, 60),
                    (163, 255, 0), (0, 82, 0), (3, 95, 161), (0, 80, 100),
                    (183, 130, 88)]
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._metainfo['dataset_type'] = "DIOR"
