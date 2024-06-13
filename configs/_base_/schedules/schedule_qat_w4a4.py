# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='exp',
    warmup_by_epoch=True,
    warmup_iters=1,
    warmup_ratio=0.001,
    # step=[9, 14])
    # step=[7, 13, 15])  # 这是属于voc ret18的
    step=[8, 14])  # 这是属于voc ret18的
runner = dict(type='EpochBasedRunner', max_epochs=16)
