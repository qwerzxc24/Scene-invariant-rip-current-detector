from ultralytics.utils import (
    DEFAULT_CFG,
    LOGGER,
    RANK,
    TQDM,
    __version__,
    callbacks,
    clean_url,
    colorstr,
    emojis,
    yaml_save,
)

from ultralytics.models import yolo
from ultralytics.nn.tasks import MambaModel_AAE

class MambaTrainer_AAE(yolo.segment.SegmentationTrainer):

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a SegmentationTrainer object with given arguments."""
        if overrides is None:
            overrides = {}
        overrides["task"] = "segment"
        super().__init__(cfg, overrides, _callbacks)
        
        
    def get_model(self, cfg=None, weights=None, verbose=True):

        model = MambaModel_AAE(
            cfg=cfg, 
            nc=self.data['nc'], 
            verbose=verbose and RANK == -1,
        )
        if weights: model.load(weights)

        if hasattr(model.model[-1], 'aae_cache'):
            model.model[-1].aae_cache = None
            
        return model
    
