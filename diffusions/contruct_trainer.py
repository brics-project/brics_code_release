"""Recover diffusion trainer
"""
import torch.nn as nn
from imagen_pytorch import Unet as Unet_Imagen
from imagen_pytorch import ImagenTrainer
from diffusions.imagen_custom import Imagen

__all__ = ['construct_imagen_trainer']

def get_layer_attns(atten_range, total_len):
    layer_attns = []

    for i in range(1, total_len+1):
        if i in atten_range:
            layer_attns.append(True)
        else:
            layer_attns.append(False)

    return tuple(layer_attns)

def construct_imagen_trainer(G, cfg, device=None, ckpt_path=None, test_flag=False):
    dim_mults = cfg.get('dim_mults', (1, 2, 2, 4))
    class_embed_dim = 512

    unet = Unet_Imagen(dim=cfg.dim,
                        text_embed_dim=class_embed_dim,
                        channels=G.feat_coord_dim,
                        dim_mults=dim_mults,
                        num_resnet_blocks=cfg.get('num_resnet_blocks', 3),
                        layer_attns=get_layer_attns(cfg.get('atten_layers', [3, 4]),
                                                    len(dim_mults)),
                        layer_cross_attns = False,
                        use_linear_attn = True,
                        cond_on_text = cfg.class_condition)
    if cfg.class_condition:
        unet.add_module("class_embedding_layer",
                        nn.Embedding(1000, class_embed_dim))
    imagen = Imagen(
            condition_on_text = cfg.class_condition,
            unets = (unet, ),
            image_sizes = (cfg.feat_spatial_size, ),
            timesteps = 1000,
            channels=G.feat_coord_dim,
            auto_normalize_img=False,
            min_snr_gamma=5,
            min_snr_loss_weight=cfg.get('use_min_snr', True),
            dynamic_thresholding=False,
            noise_schedules=cfg.get('noise_scheduler', 'cosine'),
            pred_objectives='noise', # noise or x_start
            loss_type='l2'
            )

    precision = None if cfg.mixed_precision == "no" else cfg.mixed_precision

    trainer = ImagenTrainer(imagen=imagen,
                            imagen_checkpoint_path=None,
                            lr=cfg.train_lr,
                            cosine_decay_max_steps=cfg.cosine_decay_max_steps, 
                            warmup_steps=cfg.warmup_steps,
                            use_ema=cfg.use_ema,
                            precision=precision
                            )

    if ckpt_path is not None:
        trainer.load(ckpt_path,
                    only_model=cfg.only_load_model if not test_flag else False)

    if device is not None:
        trainer = trainer.to(device)

    return trainer
