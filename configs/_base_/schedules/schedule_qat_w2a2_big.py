# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='exp',
    warmup_by_epoch=True,
    warmup_iters=1,
    warmup_ratio=0.004,
    gamma=0.5,
    step=[8, 13, 14, 15, 16, 17])  # 0.01 0.005 0.0025 0.00125 0.000625 
runner = dict(type='EpochBasedRunner', max_epochs=18)
