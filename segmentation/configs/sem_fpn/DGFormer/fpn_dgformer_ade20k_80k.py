_base_ = [
    '../../_base_/models/fpn_r50.py',
    '../../_base_/datasets/ade20k.py',
    '../../_base_/default_runtime.py'
]
# model settings
model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/dgformer.pth',
    backbone=dict(
        type='DGFormer_DGA_EFFN',
        style='pytorch'),
    neck=dict(in_channels=[32, 64, 160, 256]),
    decode_head=dict(num_classes=150))


gpu_multiples = 2  # we use 8 gpu instead of 4 in mmsegmentation, so lr*2 and max_iters/2
# optimizer
optimizer = dict(type='AdamW', lr=0.0001*gpu_multiples, weight_decay=0.0001)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=0.0, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=160000//gpu_multiples)
checkpoint_config = dict(by_epoch=False, interval=16000//gpu_multiples)
# evaluation = dict(interval=16000//gpu_multiples, metric='mIoU')
evaluation = dict(interval=80000, metric='mIoU')

