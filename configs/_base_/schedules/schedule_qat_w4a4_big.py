# 用来分析到底是训练参数的问题吗
# optimizer
optimizer = dict(type='SGD', lr=0.004, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='exp',
    warmup_by_epoch=True,
    warmup_iters=1,
    warmup_ratio=0.01,
    step=[8, 14])
runner = dict(type='EpochBasedRunner', max_epochs=16)
