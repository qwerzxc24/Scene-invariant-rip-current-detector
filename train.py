from ultralytics.models.yolo.world.mamba_train import MambaTrainer_AAE
from ultralytics import YOLO


trainer = MambaTrainer_AAE(overrides={
        'model': './ultralytics/cfg/models/mamba-yolo/yolo-mamba-seg.yaml',
        'data': '',      
        'epochs': 300,          
        'imgsz': 640,          
        'batch': 4,           
        'device': '0',         
        'project': '',
        'name': f''  ,
        'workers': 10,
        'exist_ok': True,       
        'patience': 30,
    })


trainer.train()

