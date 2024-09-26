import models


def factory(cfg):
    if cfg.model.name == 'DGCNN':
        return models.NetTwoKNN(cfg)
    if cfg.model.name == 'DGCNN2':
        return models.NetOnlyFirstKNN(cfg) 
    if cfg.model.name == 'DGCNN3':
        return models.NetOnlyFirstKNN3Layers(cfg)
    if cfg.model.name == 'NVS2GRAPH':
        return models.NetNVS2Graph(cfg)  
    if cfg.model.name == 'EST':
        return models.NetEST(cfg)            
    else:
        raise NotImplementedError(f'Model {cfg.model.name}')
