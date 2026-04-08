import argparse
import os
import importlib.util

import time
from contextlib import nullcontext

from copy import deepcopy

def _check_runtime_dependencies():
    required = {
        "numpy": "numpy",
        "torch": "torch",
        "torchvision": "torchvision",
        "PIL": "pillow",
        "tqdm": "tqdm",
        "open_clip": "open_clip_torch",
    }
    missing = [pip_name for module_name, pip_name in required.items() if importlib.util.find_spec(module_name) is None]
    if missing:
        cmd = "python3 -m pip install " + " ".join(sorted(set(missing)))
        raise SystemExit(
            "Missing required dependencies: {}\n"
            "Install them with:\n  {}".format(", ".join(sorted(set(missing))), cmd)
        )


_check_runtime_dependencies()

from PIL import Image
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets


try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import torchvision.models as models

from clip.custom_clip import get_coop
from clip.cocoop import get_cocoop
from data.imagnet_prompts import imagenet_classes
from data.datautils import AugMixAugmenter, build_dataset
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, load_model_weight, set_random_seed
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets
from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask

from mta import solve_mta

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def detect_colab_environment():
    if "COLAB_GPU" in os.environ:
        return True
    try:
        import google.colab  # type: ignore
        return True
    except Exception:
        return False


def download_cifar10_for_colab(data_root):
    cifar_root = os.path.join(data_root, "cifar10")
    os.makedirs(cifar_root, exist_ok=True)
    print(f"[colab] Downloading CIFAR10 to {cifar_root} ...")
    datasets.CIFAR10(root=cifar_root, train=True, download=True)
    datasets.CIFAR10(root=cifar_root, train=False, download=True)
    print("[colab] CIFAR10 download complete. This is for quick environment validation.")


def resolve_data_root(data_root):
    placeholder_tokens = [
        "/path/to/data",
        "/absolute/path/to/your/datasets",
        "/REAL/PATH/TO/DATASET_ROOT",
        "/your/real/path",
    ]
    if any(token in data_root for token in placeholder_tokens):
        raise SystemExit(
            "Please provide real dataset path.\n"
            "Examples:\n"
            "  --data /path/to/ImageNet\n"
            "  --data /data/ImageNet\n"
            "  --data /content/datasets/ImageNet"
        )
    return data_root


def _list_dirs_for_hint(path):
    if not path or not os.path.isdir(path):
        return []
    try:
        return sorted([d.name for d in os.scandir(path) if d.is_dir()])[:20]
    except OSError:
        return []


def select_confident_samples(logits, top):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
    return logits[idx], idx

def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True) # logits = outputs.log_softmax(dim=1) [N, 1000]
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0]) # avg_logits = logits.mean(0) [1, 1000]
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)

def mean_pool_logits(model, images):
    amp_ctx = torch.cuda.amp.autocast if torch.cuda.is_available() else nullcontext
    with torch.no_grad():
        with amp_ctx():
            logits = model(images)
    return logits.mean(dim=0, keepdim=True)


def test_time_tuning(model, inputs, optimizer, scaler, args):
    if args.cocoop:
        image_feature, pgen_ctx = inputs
        pgen_ctx.requires_grad = True
        optimizer = torch.optim.AdamW([pgen_ctx], args.lr)
    
    selected_idx = None
    amp_ctx = torch.cuda.amp.autocast if torch.cuda.is_available() else nullcontext
    for j in range(args.tta_steps):
        with amp_ctx():
            if args.cocoop:
                output = model((image_feature, pgen_ctx))
            else:
                output = model(inputs) 

            if selected_idx is not None:
                output = output[selected_idx]
            else:
                output, selected_idx = select_confident_samples(output, args.selection_p)

            loss = avg_entropy(output)
        
        optimizer.zero_grad()
        # compute gradient and do SGD step
        scaler.scale(loss).backward()
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.step(optimizer)
        scaler.update()
    if args.cocoop:
        return pgen_ctx

    return


def main(args=None):
    if args is None:
        args = parser.parse_args()
    if not args.data:
        print("[data][warning] --data not provided. Running in default test mode with synthetic data.")
        args.use_fallback_dataset = True
        args.data = "/tmp/mta_dummy_data"
    args.data = resolve_data_root(args.data)
    if not os.path.isdir(args.data):
        parent = os.path.dirname(args.data) if args.data else ""
        visible = _list_dirs_for_hint(parent)
        if not args.use_fallback_dataset:
            raise SystemExit(
                "Invalid --data path: {}\n"
                "Available directories under '{}': {}\n"
                "Suggestion: provide a real dataset root with ImageNet/val or val/test folders.".format(
                    args.data,
                    parent if parent else "(unknown)",
                    ", ".join(visible) if visible else "(none found)"
                )
            )
        print(f"[data][warning] Provided path does not exist: {args.data}")
        print("[data][warning] Using fallback synthetic dataset mode.")

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[device][warning] CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"

    if detect_colab_environment():
        print("[colab] Google Colab environment detected.")
        print("[colab] Install deps: !pip install ftfy regex tqdm scipy")
        if args.colab_download_cifar:
            download_cifar10_for_colab(args.data)

    set_random_seed(args.seed)
    main_worker(args.gpu, args)


def main_worker(gpu, args):
    args.gpu = gpu
    set_random_seed(args.seed)
    if args.device == "cuda":
        if args.gpu is None:
            args.gpu = 0
        device = torch.device(f"cuda:{args.gpu}")
    else:
        args.gpu = None
        device = torch.device("cpu")
    print("Using device: {}".format(device))

    if args.mta:
        if args.arch not in {"ViT-B/16", "ViT-L/14"}:
            raise ValueError(
                "MTA requires a CLIP ViT backbone (ViT-B/16 minimum, ViT-L/14 preferred). "
                f"Received: {args.arch}"
            )
        if args.mta_tau <= 0:
            raise ValueError("mta_tau must be > 0 for temperature-scaled text affinity.")
        if not (64 <= args.mta_views <= 128):
            print(
                "Warning: mta_views is outside the paper-aligned range [64, 128]. "
                f"Current value: {args.mta_views}"
            )

    # create model (zero-shot clip model (ViT-L/14@px336) with promptruning)
    if args.test_sets in fewshot_datasets:
        classnames = eval("{}_classes".format(args.test_sets.lower()))
    else:
        classnames = imagenet_classes
    if args.cocoop:
        model = get_cocoop(args.arch, args.test_sets, 'cpu', args.n_ctx)
        assert args.load is not None
        load_model_weight(args.load, model, 'cpu', args) # to load to cuda: device="cuda:{}".format(args.gpu)
        model_state = deepcopy(model.state_dict())
    else:
        model = get_coop(args.arch, args.test_sets, str(device), args.n_ctx, args.ctx_init)
        if args.load is not None:
            print("Use pre-trained soft prompt (CoOp) as initialization")
            pretrained_ctx = torch.load(args.load)['state_dict']['ctx']
            assert pretrained_ctx.size()[0] == args.n_ctx
            with torch.no_grad():
                model.prompt_learner[0].ctx.copy_(pretrained_ctx)
                model.prompt_learner[0].ctx_init_state = pretrained_ctx
        model_state = None

    for name, param in model.named_parameters():
        if not args.cocoop:
            if "prompt_learner" not in name:
                param.requires_grad_(False)
        else:
            if "text_encoder" not in name:
                param.requires_grad_(False)
    
    print("=> Model created: visual backbone {}".format(args.arch))
    
    if device.type == "cuda":
        torch.cuda.set_device(args.gpu)
    model = model.to(device)

    # define optimizer
    if args.cocoop or args.mta:
        optimizer = None
        optim_state = None
    else:
        trainable_param = model.prompt_learner.parameters()
        optimizer = torch.optim.AdamW(trainable_param, args.lr)
        optim_state = deepcopy(optimizer.state_dict())

    # setup automatic mixed-precision (Amp) loss scaling
    scaler = torch.cuda.amp.GradScaler(init_scale=1000, enabled=(device.type == "cuda"))

    print('=> Using native Torch AMP. Training in mixed precision.')

    cudnn.benchmark = True

    # norm stats from clip.load()
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])

    
    # iterating through eval datasets
    datasets = args.test_sets.split("/")
    results = {}
    for set_id in datasets:
        if args.tpt or args.mta:
            base_transform = transforms.Compose([
                transforms.Resize(args.resolution, interpolation=BICUBIC),
                transforms.CenterCrop(args.resolution)])
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                normalize])
            n_views = args.mta_views if args.mta else args.batch_size - 1
            data_transform = AugMixAugmenter(
                base_transform,
                preprocess,
                n_views=n_views,
                augmix=(len(set_id)>1 and args.tpt),
                use_mta_ops=args.mta
            )
            batchsize = 1
        else:
            data_transform = transforms.Compose([
                transforms.Resize(args.resolution, interpolation=BICUBIC),
                transforms.CenterCrop(args.resolution),
                transforms.ToTensor(),
                normalize,
            ])
            batchsize = args.batch_size

        print("evaluating: {}".format(set_id))
        # reset the model
        # Reset classnames of custom CLIP model
        if len(set_id) > 1: 
            # fine-grained classification datasets
            classnames = eval("{}_classes".format(set_id.lower()))
        else:
            assert set_id in ['A', 'R', 'K', 'V', 'I']
            classnames_all = imagenet_classes
            classnames = []
            if set_id in ['A', 'R', 'V']:
                label_mask = eval("imagenet_{}_mask".format(set_id.lower()))
                if set_id == 'R':
                    for i, m in enumerate(label_mask):
                        if m:
                            classnames.append(classnames_all[i])
                else:
                    classnames = [classnames_all[i] for i in label_mask]
            else:
                classnames = classnames_all
        if args.cocoop:
            model.prompt_generator.reset_classnames(classnames, args.arch)
            model = model.cpu()
            model_state = model.state_dict()
            model = model.to(device)
        else:
            model.reset_classnames(classnames, args.arch)

        val_dataset = build_dataset(
            set_id,
            data_transform,
            args.data,
            mode=args.dataset_mode,
            allow_fallback=args.use_fallback_dataset
        )
        print("number of test samples: {}".format(len(val_dataset)))
        val_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=batchsize, shuffle=True,
                    num_workers=args.workers, pin_memory=(device.type == "cuda"))
            
        results[set_id] = test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, args)
        del val_dataset, val_loader
        try:
            print("=> Acc. on testset [{}]: @1 {}/ @5 {}".format(set_id, results[set_id][0], results[set_id][1]))
        except:
            print("=> Acc. on testset [{}]: {}".format(set_id, results[set_id]))

    print("======== Result Summary ========")
    print("params: nstep	lr	bs")
    print("params: {}	{}	{}".format(args.tta_steps, args.lr, args.batch_size))
    print("\t\t [set_id] \t\t Top-1 acc. \t\t Top-5 acc.")
    for id in results.keys():
        print("{}".format(id), end="	")
    print("\n")
    for id in results.keys():
        print("{:.2f}".format(results[id][0]), end="	")
    print("\n")


def test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, args):
    device = next(model.parameters()).device
    amp_ctx = torch.cuda.amp.autocast if device.type == "cuda" else nullcontext

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    baseline_top1 = AverageMeter('Base@1', ':6.2f', Summary.AVERAGE)
    baseline_top5 = AverageMeter('Base@5', ':6.2f', Summary.AVERAGE)
    mean_top1 = AverageMeter('Mean@1', ':6.2f', Summary.AVERAGE)
    mean_top5 = AverageMeter('Mean@5', ':6.2f', Summary.AVERAGE)

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')

    # reset model and switch to evaluate mode
    model.eval()
    if not args.cocoop and not args.mta: # no need to reset cocoop because it's fixed
        with torch.no_grad():
            model.reset()
    end = time.time()
    for i, (images, target) in enumerate(val_loader):
        if isinstance(images, list):
            for k in range(len(images)):
                images[k] = images[k].to(device, non_blocking=(device.type == "cuda"))
            image = images[0]
        else:
            if len(images.size()) > 4:
                # when using ImageNet Sampler as the dataset
                assert images.size()[0] == 1
                images = images.squeeze(0)
            images = images.to(device, non_blocking=(device.type == "cuda"))
            image = images
        target = target.to(device, non_blocking=(device.type == "cuda"))
        if args.tpt or args.mta:
            images = torch.cat(images, dim=0)

        # reset the tunable prompt to its initial state
        if not args.cocoop: # no need to reset cocoop because it's fixed
            if args.tpt and args.tta_steps > 0:
                with torch.no_grad():
                    model.reset()
            if args.tpt:
                optimizer.load_state_dict(optim_state)
                test_time_tuning(model, images, optimizer, scaler, args)
            elif args.mta:
                if args.eval_mta_variants:
                    with torch.no_grad():
                        with amp_ctx():
                            base_output = model(image)
                    mean_output = mean_pool_logits(model, images)
                    b1, b5 = accuracy(base_output, target, topk=(1, 5))
                    m1, m5 = accuracy(mean_output, target, topk=(1, 5))
                    baseline_top1.update(b1[0], image.size(0))
                    baseline_top5.update(b5[0], image.size(0))
                    mean_top1.update(m1[0], image.size(0))
                    mean_top5.update(m5[0], image.size(0))
                output = solve_mta(model, images, args)
        else:
            with torch.no_grad():
                with amp_ctx():
                    image_feature, pgen_ctx = model.gen_ctx(images, args.tpt)
            optimizer = None
            pgen_ctx = test_time_tuning(model, (image_feature, pgen_ctx), optimizer, scaler, args)

        # The actual inference goes here
        if args.tpt:
            if args.cocoop:
                image_feature = image_feature[0].unsqueeze(0)
        
        if not args.mta:
            with torch.no_grad():
                with amp_ctx():
                    if args.cocoop:
                        output = model((image_feature, pgen_ctx))
                    else:
                        output = model(image)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
                
        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0:
            progress.display(i)

    progress.display_summary()
    if args.mta and args.eval_mta_variants:
        print(
            "MTA variants @1: baseline {:.2f} | mean-pool {:.2f} | mta {:.2f} | +{:.2f} vs baseline | +{:.2f} vs mean".format(
                baseline_top1.avg, mean_top1.avg, top1.avg,
                top1.avg - baseline_top1.avg,
                top1.avg - mean_top1.avg
            )
        )

    return [top1.avg, top5.avg]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test-time Prompt Tuning')
    parser.add_argument('--data', type=str, help='path to dataset root')
    parser.add_argument('--test_sets', type=str, default='A/R/V/K/I', help='test dataset (multiple datasets split by slash)')
    parser.add_argument('--dataset_mode', type=str, default='test', help='which split to use: train/val/test')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='ViT-L/14')
    parser.add_argument('--resolution', default=224, type=int, help='CLIP image resolution')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N')
    parser.add_argument('--lr', '--learning-rate', default=5e-3, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('-p', '--print-freq', default=200, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--device', type=str, default=("cuda" if torch.cuda.is_available() else "cpu"),
                        choices=["cuda", "cpu"], help='execution device')
    parser.add_argument('--tpt', action='store_true', default=False, help='run test-time prompt tuning')
    parser.add_argument('--selection_p', default=0.1, type=float, help='confidence selection percentile')
    parser.add_argument('--tta_steps', default=1, type=int, help='test-time-adapt steps')
    parser.add_argument('--n_ctx', default=4, type=int, help='number of tunable tokens')
    parser.add_argument('--ctx_init', default='a_photo_of_a', type=str, help='init tunable prompts')
    parser.add_argument('--cocoop', action='store_true', default=False, help="use cocoop's output as prompt initialization")
    parser.add_argument('--load', default=None, type=str, help='path to a pre-trained coop/cocoop')
    parser.add_argument('--seed', type=int, default=0)
    
    # MTA arguments
    parser.add_argument('--mta', action='store_true', default=False, help='run meanshift test-time adaptation (MTA)')
    parser.add_argument('--lambda_q', default=4.0, type=float, help='quadratic term weighting factor')
    parser.add_argument('--lambda_y', default=0.2, type=float, help='entropic term weighting factor')
    parser.add_argument('--mta_tau', default=1.0, type=float, help='temperature for text-affinity in MTA (must be > 0)')
    parser.add_argument('--mta_max_iter', default=20, type=int, help='max alternating iterations for MTA')
    parser.add_argument('--mta_tol', default=1e-6, type=float, help='convergence threshold for MTA')
    parser.add_argument('--mta_bandwidth_frac', default=0.3, type=float, help='fraction of neighbors for adaptive bandwidth')
    parser.add_argument('--mta_views', default=127, type=int, help='number of augmented views for MTA')
    parser.add_argument('--eval_mta_variants', action='store_true', default=False, help='report baseline and mean-pooling vs MTA')
    parser.add_argument('--use_fallback_dataset', action='store_true', help='fallback to FakeData if dataset path is invalid')
    parser.add_argument('--no_fallback_dataset', action='store_true', help='disable fallback dataset behavior')
    parser.add_argument('--colab_download_cifar', action='store_true', help='download CIFAR10 in Colab for quick setup validation')
    parser.set_defaults(use_fallback_dataset=False)
    
    cli_args = parser.parse_args()
    if cli_args.no_fallback_dataset:
        cli_args.use_fallback_dataset = False
    main(cli_args)