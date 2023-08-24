_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
# optimizer
model = dict(
    pretrained='pretrained/dgformer.pth',
    backbone=dict(
        type='DGFormer_DGA_EFFN',
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[32, 64, 160, 256],
        out_channels=256,
        num_outs=5))
# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0002, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
