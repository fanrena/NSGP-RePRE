# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=9, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=9,
        by_epoch=True,
        milestones=[7, 9],
        gamma=0.1)
]


optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGDNSCL', lr=0.02, momentum=0.9, weight_decay=0.0001, svd=True),
    # paramwise_cfg=dict(norm_decay_mult=0., bypass_duplicate=True)
)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
