
import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
import pdb
import utils
import vision_transformer as vits
from vision_transformer import DINOHead
from einops import rearrange, repeat
import platform
torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

def get_args_parser():
    parser = argparse.ArgumentParser('LAFS', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='mynet', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'] \
                + torchvision_archs + torch.hub.list("facebookresearch/xcit:main"),
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=8, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=100000, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")#65536
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=82, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')#64   #a5000 260, na:130 pandora   ,#large 46 on pandora, 82 on A100
    parser.add_argument('--epochs', default=41, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument('--data_path', default='/data/scratch/acw569/webface_4m/ms1mv3/ms1m-retinaface-t1', type=str,
        help='Please specify path to the Face training data, using MXNet format, i.e. .rec file.')
        #landmark_path
    parser.add_argument('--landmark_path', default='/data/home/acw569/precheck/ms1m_196_land_sp/Backbone_VIT_land_8_Epoch_34_Batch_523881_Time_2021-07-31-11-07_checkpoint.pth', type=str,
        help='Please specify path to the Pretrained landmark CNN.')
    parser.add_argument('--output_dir', default="/data/scratch/acw569/checkpoint/ssl/ms1m_land_ms1mland_1m_40epoch_noflip", type=str, 
                help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=10, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=6, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="tcp://127.0.0.1:12585", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    return parser

def load_part_checkpoint_landmark(path,model,pretrain_name=['stn','output']):
    # pdb.set_trace()
    pretrained_dict =  torch.load(path, map_location='cpu')
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    # pretrained_dict=list(pretrained_dict.keys())
    back_remove=list(pretrained_dict.keys())
    for keys in back_remove:
        if 'dummy_orthogonal_classifier' in keys:
            # pdb.set_trace()
            continue
        pretrained_dict[keys.replace('module.','')]=pretrained_dict.pop(keys)

    # pdb.set_trace()
    # for name_space in pretrain_name:
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if pretrain_name[0] in k or pretrain_name[1] in k}
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if pretrain_name[0] in k or pretrain_name[1] in k}
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    model.load_state_dict(model_dict,strict=True)
    # model.encoder.output_layer.load_state_dict(pretrained_dict,strict=True)
    model_dict = model.state_dict()
    #freeze stn and output layer
    for name, param in model.named_parameters():
        # if not param.requires_grad:
        if pretrain_name[0] in name or pretrain_name[1] in name:
            # pdb.set_trace()
            param.requires_grad = False

def train_lafs(args):
    data_path = args.data_path
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    transform = DataAugmentation_LAFS(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
    )
    
    from face_pre_pro.dataloader_web import FaceDataset
    
    dataset = FaceDataset(
        os.path.join(data_path, 'train.rec'), 
        dino_trans=transform,
        rand_mirror=False,
        random_resizecrop=False,
        rand_au=False,
        sifenzhiyi=True,
        filepath_id_nidex='ms1m_random_index.json'
    )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building networks ... ============
    from face_pre_pro.ViT_face import ViT_face_landmark_patch8, face_landmark_4simmin_glo_loc, HybridFusion
    from face_pre_pro.iresnet import iresnet50
    
    knowledge_dis = True
    
    # ========== Solution 2: Add ResNet Backbone ==========
    print("Building ResNet50 backbone for hybrid fusion...")
    resnet_backbone = iresnet50(
        dropout=0.1, 
        fp16=False, 
        num_features=512,
        NUM_CLASS=30000
    )
    resnet_backbone = resnet_backbone.cuda()
    
    # Freeze ResNet (we only use it for feature extraction)
    for param in resnet_backbone.parameters():
        param.requires_grad = False
    resnet_backbone.eval()
    print("✓ ResNet50 loaded and frozen")
    # =====================================================
    
    if knowledge_dis:
        '''Load pretrained landmark CNN'''
        landmarkcnn = face_landmark_4simmin_glo_loc(
            loss_type='CosFace',
            GPU_ID=None,
            num_class=300,
            num_patches=196,
            image_size=112,
            patch_size=8,
            dim=512,
            depth=12,
            heads=11,
            mlp_dim=2560,
            dropout=0.1,
            emb_dropout=0.1
        )
        landmarkcnn = landmarkcnn.cuda()
        load_part_checkpoint_landmark(
            path=args.landmark_path,
            model=landmarkcnn,
            pretrain_name=['stn', 'output']
        )
        landmarkcnn.eval()
        print("✓ Landmark CNN loaded")

    # Build student and teacher (Part fViT)
    teacher = ViT_face_landmark_patch8(
        loss_type='CosFace',
        GPU_ID=None,
        num_class=30000,
        image_size=112,
        patch_size=8,
        dim=768,
        depth=12,
        heads=11,
        num_patches=196,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1,
        with_land=False,
        use_standcoord=False,
        Random_prob=False,
        shuffle=False
    )
    
    student = ViT_face_landmark_patch8(
        loss_type='CosFace',
        GPU_ID=None,
        num_class=30000,
        image_size=112,
        patch_size=8,
        dim=768,
        depth=12,
        heads=11,
        num_patches=196,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1,
        with_land=False,
        use_standcoord=False,
        Random_prob=False,
        shuffle=False
    )

    embed_dim = 768
    
    # ========== Solution 2: Add Hybrid Fusion Module ==========
    print("Building Hybrid Fusion module...")
    fusion_module = HybridFusion(
        vit_dim=768,      # Part fViT output dimension
        resnet_dim=512,   # ResNet output dimension
        out_dim=768,      # Keep same as ViT for DINO head
        num_heads=8
    )
    fusion_module = fusion_module.cuda()
    print("✓ Hybrid Fusion module created")
    # ===========================================================

    # Multi-crop wrapper with DINO head
    student = utils.MultiCropWrapper(
        student, 
        DINOHead(
            embed_dim,
            args.out_dim,
            use_bn=args.use_bn_in_head,
            norm_last_layer=args.norm_last_layer,
        )
    )
    teacher = utils.MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
    )

    # Move to GPU
    student, teacher = student.cuda(), teacher.cuda()
    landmarkcnn = landmarkcnn.cuda()
    
    # Synchronize batch norms
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)
        
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
        landmarkcnn = nn.parallel.DistributedDataParallel(landmarkcnn, device_ids=[args.gpu])
    else:
        teacher_without_ddp = teacher
        
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    
    # Teacher and student start with same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    
    # No gradients for teacher
    for p in teacher.parameters():
        p.requires_grad = False
    for p in landmarkcnn.parameters():
        p.requires_grad = False
    # ========== Solution 2: No gradients for ResNet ==========
    for p in resnet_backbone.parameters():
        p.requires_grad = False
    # ==========================================================
    
    print(f"Student and Teacher are built: Part fViT network with Hybrid Fusion")

    # ============ preparing loss ... ============
    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
        enable_dynamic_weighting=True  # Solution 3
    ).cuda()

    # ============ preparing optimizer ... ============
    # ========== Solution 2: Add fusion module parameters ==========
    params_groups = utils.get_params_groups(student)
    
    # Add fusion module parameters to optimizer
    fusion_params = {'params': fusion_module.parameters(), 'weight_decay': 0.04}
    params_groups.append(fusion_params)
    # ===============================================================
    
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)
        
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    momentum_schedule = utils.cosine_scheduler(
        args.momentum_teacher, 1,
        args.epochs, len(data_loader)
    )
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting DINO training with Hybrid Fusion!")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch ... ============
        train_stats = train_one_epoch_hybrid(  # ✅ Use hybrid version
            student, teacher, teacher_without_ddp, landmarkcnn, 
            resnet_backbone, fusion_module,  # Include new modules
            dino_loss, data_loader, optimizer, 
            lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args
                                        )

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
            'fusion_module': fusion_module.state_dict(),  # Save fusion weights
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
                
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
def train_one_epoch_hybrid(student, teacher, teacher_without_ddp, landmarkcnn, 
                          resnet_backbone, fusion_module,  # New parameters
                          dino_loss, data_loader, optimizer, 
                          lr_schedule, wd_schedule, momentum_schedule, 
                          epoch, fp16_scaler, args):
    """
    Training loop with Hybrid Fusion (Solution 2).
    Combines Part fViT and ResNet features via cross-attention.
    """
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    
    ori_image_index = np.array([0, 3, 5, 7, 9, 11, 13, 15, 17])
    aug_image_index = ori_image_index + 1
    
    for it, (images, _) in enumerate(metric_logger.log_every(data_loader, 100, header)):
        it = len(data_loader) * epoch + it
        
        # Update learning rate and weight decay
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:
                param_group["weight_decay"] = wd_schedule[it]

        # Move images to GPU
        images = [im.cuda(non_blocking=True) for im in images]
        
        # ========== Extract landmark features (existing code) ==========
        land_label, img_reconstructed = landmarkcnn(
            images[0], 
            x_Aug=images[1], 
            Random_prob=True, 
            return_prob=True, 
            random_coor=False
        )
        
        img_emb0 = rearrange(
            img_reconstructed, 
            'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
            p1=landmarkcnn.patch_size, 
            p2=landmarkcnn.patch_size
        )
        
        land_label, img_reconstructed = landmarkcnn(
            images[2], 
            x_Aug=images[3], 
            Random_prob=True, 
            return_prob=True, 
            random_coor=False
        )
        
        img_emb1 = rearrange(
            img_reconstructed, 
            'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
            p1=landmarkcnn.patch_size, 
            p2=landmarkcnn.patch_size
        )
        
        # Local views
        images_to_prob = torch.stack(images[4::2])
        loc, b, c, w, h = images_to_prob.shape
        images_to_prob = images_to_prob.view(loc * b, c, w, h)
        
        images_to_prob_aug = torch.stack(images[5::2])
        images_to_prob_aug = images_to_prob_aug.view(loc * b, c, w, h)
        
        land_label, img_reconstructed_loc = landmarkcnn(
            images_to_prob, 
            x_Aug=images_to_prob_aug, 
            Random_prob=True, 
            ran_sample=True, 
            random_coor=False
        )
        
        img_emb2 = rearrange(
            img_reconstructed_loc, 
            'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
            p1=landmarkcnn.patch_size, 
            p2=landmarkcnn.patch_size
        )
        img_emb2 = img_emb2.view(loc, b, 36, 192)
        img_emb2 = [emg for emg in img_emb2]
        
        images = [img_emb0] + [img_emb1] + img_emb2
        # ================================================================
        
        # ========== Solution 2: Extract ResNet Features ==========
        with torch.no_grad():  # ResNet is frozen, no gradients needed
            # Only process global views for ResNet (first 2 views)
            global_images_for_resnet = torch.cat([
                images[0][:, :, :].mean(dim=1),  # Average over patches -> [B, C]
                images[1][:, :, :].mean(dim=1)
            ], dim=0)  # [2B, C]
            
            # Convert embeddings back to images for ResNet
            # ResNet expects [B, 3, 112, 112], but we have embeddings
            # So we'll use the original images before landmark processing
            resnet_input = torch.cat([images[0], images[1]], dim=0) / 255.0 * 2 - 1
            resnet_features = resnet_backbone(resnet_input)  # [2B, 512]
        # ===========================================================
        
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            # Teacher and student forward passes (Part fViT)
            teacher_output_vit = teacher(images[:2])  # [2B, out_dim]
            student_output_vit = student(images)      # [(2+local)B, out_dim]
            
            # ========== Solution 2: Apply Hybrid Fusion ==========
            # Fuse teacher features with ResNet
            teacher_output_fused = fusion_module(
                teacher_output_vit,  # [2B, 768]
                resnet_features      # [2B, 512]
            )
            # =====================================================
            
            # Compute DINO loss with fused teacher features
            current_batch_size = images[0].size(0)
            batch_ratio = current_batch_size / len(data_loader.dataset)
            
            loss = dino_loss(
                student_output_vit,    # Student sees only ViT features
                teacher_output_fused,  # Teacher uses fused features
                epoch,
                batch_ratio=batch_ratio
            )

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # Student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for teacher
        with torch.no_grad():
            m = momentum_schedule[it]
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # Logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train_one_epoch(student, teacher, teacher_without_ddp,landmarkcnn, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    # print(header)
    soft_split = nn.Unfold(kernel_size=(10, 10), stride=(9, 9), padding=(1, 1))
    # import kornia as K
    # pdb.set_trace()
    # flip_and_color_jitter=K.augmentation.container.AugmentationSequential(
    #     K.augmentation.RandomHorizontalFlip(p=0.5),
    #     K.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
    #     K.augmentation.RandomGrayscale(p=0.2)

    # )
    # # GaussianBlur=K.augmentation.RandomGaussianBlur((3, 3),sigma=(0.1,2),p=0.5)
    # # Solarization=K.augmentation.RandomSolarize(p=0.2)

    # global_transfo1=K.augmentation.container.AugmentationSequential(
    #     flip_and_color_jitter,
    #     K.augmentation.RandomGaussianBlur((3, 3),sigma=(0.1,2),p=1.0)
    # )
    # # global_transfo2=K.AugmentationSequential(
    # #     flip_and_color_jitter,
    # #     K.augmentation.RandomGaussianBlur((3, 3),sigma=(0.1,2),p=0.1),
    # #     K.augmentation.RandomSolarize(p=0.2)
    # # )
    # local_transfo=K.augmentation.container.AugmentationSequential(
    #     flip_and_color_jitter,
    #     K.augmentation.RandomGaussianBlur((3, 3),sigma=(0.1,2),p=0.5),
    #     # K.augmentation.RandomSolarize(p=0.2)
    # )
    ori_image_index=np.array([0,3,5,7,9,11,13,15,17])
    aug_image_index=ori_image_index+1#[1,3,5,7,9,11,13,15,17]
    for it, (images, _) in enumerate(metric_logger.log_every(data_loader, 100, header)):
        # if it%10==0:
        #     print(it)
        # pdb.set_trace()
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        # pdb.set_trace()
        images = [im.cuda(non_blocking=True) for im in images]#*2 -1 converge better?
        # pdb.set_trace()
        # images = images.cuda(non_blocking=True).float()
        # images = [im.to(self.args.device)*2-1 for im in images]

        #landmark branch
        # pdb.set_trace()
        embd_list=[]
        # pdb.set_trace()
        # images_glo1_aug=global_transfo1(images[0])
        # images[0]=images[0]/255.0*2-1
        # images_glo1_aug=images_glo1_aug/255.0*2-1
        # land_label,img_reconstructed=landmarkcnn(images[0],images_glo1_aug)
        # pdb.set_trace()
        land_label,img_reconstructed=landmarkcnn(images[0],x_Aug=images[1],Random_prob=True,return_prob=True,random_coor=False)

        #reconstructed image to embedding    
        img_emb0 = rearrange(img_reconstructed, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = landmarkcnn.patch_size, p2 = landmarkcnn.patch_size)# no landmark on glo
        # img_emb0 = rearrange(images[1], 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = landmarkcnn.patch_size, p2 = landmarkcnn.patch_size)
        #
        land_label,img_reconstructed=landmarkcnn(images[2],x_Aug=images[3],Random_prob=True,return_prob=True,random_coor=False)

        #reconstructed image to embedding
        img_emb1 = rearrange(img_reconstructed, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = landmarkcnn.patch_size, p2 = landmarkcnn.patch_size)
        # img_emb1 = rearrange(images[3], 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = landmarkcnn.patch_size, p2 = landmarkcnn.patch_size)
        # img_emb1=[soft_split(img_view).transpose(1,2) for img_view in images[2:]]#views+ b,patches, 300   image 1 #index=2
        # pdb.set_trace()

        #local views 
        # pdb.set_trace()
        # images_to_prob=torch.stack(images[3:])#torch.from_numpy(images[2:])
        # pdb.set_trace()
        images_to_prob=torch.stack(images[4::2])
        loc,b,c,w,h=images_to_prob.shape
        images_to_prob=images_to_prob.view(loc*b,c,w,h)
        #aug image
        images_to_prob_aug=torch.stack(images[5::2])
        loc,b,c,w,h=images_to_prob_aug.shape
        images_to_prob_aug=images_to_prob_aug.view(loc*b,c,w,h)
        # images_to_prob_Aug=local_transfo(images_to_prob)# prob
        # images_to_prob=images_to_prob/255.0*2 -1
        # images_to_prob_Aug=images_to_prob_Aug/255.0*2 -1
        # land_label,img_reconstructed_loc=landmarkcnn(images_to_prob,images_to_prob_Aug,Random_prob=True)# maybe need to be reshape
        # pdb.set_trace()
        land_label,img_reconstructed_loc=landmarkcnn(images_to_prob,x_Aug=images_to_prob_aug,Random_prob=True,ran_sample=True,random_coor=False)# maybe need to be reshape
        img_emb2 = rearrange(img_reconstructed_loc, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = landmarkcnn.patch_size, p2 = landmarkcnn.patch_size)
        # pdb.set_trace()
        img_emb2=img_emb2.view(loc,b,36,192)#.tolist()
        img_emb2=[emg for emg in img_emb2]
        # rand_sample=torch.randint(0, 144, (loc,b,25))# save 25 patches
        # img_emb2=img_emb2.view(loc,b,144,300)
        # img_emb2=img_emb2[rand_sample]# extract 25 patches for
        # images=[img_emb0]+[img_emb1[0]]+img_emb2#.tolist()
        images=[img_emb0]+[img_emb1]+img_emb2#.tolist()
        # images_glo = torch.cat([img_emb0,img_emb1], dim=0)
        # teacher and student forward passes + compute dino loss
    with torch.cuda.amp.autocast(fp16_scaler is not None):
        teacher_output = teacher(images[:2])
        student_output = student(images)
    
    # Calculate batch ratio for dynamic weighting
        current_batch_size = images[0].size(0)  # First view batch size
        total_dataset_size = len(data_loader.dataset)
        batch_ratio = current_batch_size / total_dataset_size
    
    # Compute loss with dynamic weighting
        loss = dino_loss(student_output, teacher_output, epoch, batch_ratio=batch_ratio)
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class DINOLoss(nn.Module):
    """
    DINO Loss with Dynamic Weighting for better large-scale data handling.
    
    Dynamic weighting adapts loss weights per batch based on subset ratio,
    giving higher weights to sparse/hard batches for improved scaling.
    """
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9, enable_dynamic_weighting=True):
        """
        Args:
            out_dim: Dimension of DINO head output
            ncrops: Total number of crops (2 global + local_crops_number)
            warmup_teacher_temp: Initial teacher temperature
            teacher_temp: Final teacher temperature after warmup
            warmup_teacher_temp_epochs: Number of warmup epochs
            nepochs: Total training epochs
            student_temp: Student temperature (default: 0.1)
            center_momentum: EMA momentum for center update (default: 0.9)
            enable_dynamic_weighting: Enable adaptive loss weighting (default: True)
        """
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.enable_dynamic_weighting = enable_dynamic_weighting
        
        # Center for teacher output (prevents collapse)
        self.register_buffer("center", torch.zeros(1, out_dim))
        
        # Teacher temperature schedule with warmup
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch, batch_ratio=None):
        """
        Compute cross-entropy loss between softmax outputs of teacher and student.
        
        Args:
            student_output: Student network output [B*ncrops, out_dim]
            teacher_output: Teacher network output [B*2, out_dim] (2 global views)
            epoch: Current training epoch
            batch_ratio: Optional ratio of current_batch_size / total_dataset_size
                        Used for dynamic weighting (set to None to disable)
        
        Returns:
            total_loss: Scalar loss value
        """
        # Split student output into separate crops
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # Teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)  # 2 global teacher views

        # ========== Dynamic Weighting (Solution 3) ==========
        if self.enable_dynamic_weighting and batch_ratio is not None:
            # Compute adaptive weight based on batch subset ratio
            # Smaller batches (harder examples) get higher weights
            if batch_ratio < 0.5:  # Only apply for small batches
                # Logarithmic scaling: smaller ratio → higher weight
                weight = 1.0 / (math.log(batch_ratio + 1e-6) + 1.0)
                # Clamp to prevent extreme values
                weight = max(0.5, min(weight, 2.0))
            else:
                weight = 1.0
        else:
            weight = 1.0
        # =====================================================

        # Compute cross-entropy loss between all student-teacher pairs
        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # Skip cases where student and teacher operate on same view
                    continue
                # Cross-entropy: -teacher * log(student)
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                # Apply dynamic weight
                total_loss += loss.mean() * weight
                n_loss_terms += 1
        
        # Average over all loss terms
        total_loss /= n_loss_terms
        
        # Update teacher center with EMA
        self.update_center(teacher_output)
        
        return total_loss

    @torch.no_grad()  
    def update_center(self, teacher_output):
        """
        Update center used for teacher output with exponential moving average.
        This prevents model collapse by centering the teacher predictions.
        
        Args:
            teacher_output: Teacher network output [B*2, out_dim]
        """
        # Compute batch center
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        
        # All-reduce across GPUs (if using distributed training)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # EMA update: center = momentum * center + (1 - momentum) * batch_center
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

class DINOLoss_land(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, teacher_land_output ,epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        teacher_out=teacher_out.detach()
        teacher_out=torch.cat([teacher_out,teacher_land_output],dim=1)#??
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.chunk(2)

        # #
        # teacher_land_output = F.softmax((teacher_land_output - self.center) / temp, dim=-1)
        # teacher_land_output = teacher_land_output.chunk(2)

        #
        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()  
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(112, scale=global_crops_scale, interpolation=Image.BICUBIC),#224
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(112, scale=global_crops_scale, interpolation=Image.BICUBIC),#224
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(48, scale=local_crops_scale, interpolation=Image.BICUBIC),#224
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops

class DataAugmentation_LAFS(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            #transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(112, scale=global_crops_scale, interpolation=Image.BICUBIC),#224#[0.8,1.0]
            transforms.RandomHorizontalFlip(p=0.5)
            # flip_and_color_jitter,
            # utils.GaussianBlur(1.0),
            # normalize,
        ])
        self.global_transfo1_con1 = transforms.Compose([
            # transforms.RandomResizedCrop(112, scale=global_crops_scale, interpolation=Image.BICUBIC),#224#[0.8,1.0]
            # flip_and_color_jitter,
            # utils.GaussianBlur(1.0),
            normalize,
        ])
        self.global_transfo1_con2 = transforms.Compose([
            # transforms.RandomResizedCrop(112, scale=global_crops_scale, interpolation=Image.BICUBIC),#224#[0.8,1.0]
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(112, scale=global_crops_scale, interpolation=Image.BICUBIC),#224#global_crops_scale
            transforms.RandomHorizontalFlip(p=0.5)
            # flip_and_color_jitter,
            # utils.GaussianBlur(0.1),
            # utils.Solarization(0.2),
            # normalize,
        ])
        self.global_transfo2_con1 = transforms.Compose([
            # transforms.RandomResizedCrop(112, scale=global_crops_scale, interpolation=Image.BICUBIC),#224#global_crops_scale
            # flip_and_color_jitter,
            # utils.GaussianBlur(0.1),
            # utils.Solarization(0.2),
            normalize,
        ])
        self.global_transfo2_con2 = transforms.Compose([
            # transforms.RandomResizedCrop(112, scale=global_crops_scale, interpolation=Image.BICUBIC),#224#global_crops_scale
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(112, scale=global_crops_scale, interpolation=Image.BICUBIC),#224#local_crops_scale
            transforms.RandomHorizontalFlip(p=0.5)
            # flip_and_color_jitter,
            # utils.GaussianBlur(p=0.5),
            # normalize,
        ])
        self.local_transfo_con1= transforms.Compose([
            # transforms.RandomResizedCrop(112, scale=local_crops_scale, interpolation=Image.BICUBIC),#224
            # flip_and_color_jitter,
            # utils.GaussianBlur(p=0.5),
            normalize,
        ])
        self.local_transfo_con2 = transforms.Compose([
            # transforms.RandomResizedCrop(112, scale=local_crops_scale, interpolation=Image.BICUBIC),#224
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        # pdb.set_trace()
        crop_img=self.global_transfo1(image)#
        crops.append(self.global_transfo1_con1(crop_img))#glo1.0
        crops.append(self.global_transfo1_con2(crop_img))#glo1.1
        #glo crop
        crop_img_glo2=self.global_transfo2(image)#glo2
        crops.append(self.global_transfo2_con1(crop_img_glo2))#glo2.0
        crops.append(self.global_transfo2_con2(crop_img_glo2))#glo2.1
        for _ in range(self.local_crops_number):
            crop_img_loc=self.local_transfo(image)
            crops.append(self.local_transfo_con1(crop_img_loc))#loc_1.0
            crops.append(self.local_transfo_con2(crop_img_loc))#loc_1.1
        return crops

if __name__ == '__main__':
    parser = argparse.ArgumentParser('LAFS', parents=[get_args_parser()])
    args = parser.parse_args()
    
    train_lafs(args)
