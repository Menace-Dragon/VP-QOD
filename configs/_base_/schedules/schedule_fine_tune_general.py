# optimizer
optimizer = dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='exp',
    warmup_by_epoch=True,
    warmup_iters=1,
    warmup_ratio=0.1,
    step=[5])
runner = dict(type='EpochBasedRunner', max_epochs=8)
