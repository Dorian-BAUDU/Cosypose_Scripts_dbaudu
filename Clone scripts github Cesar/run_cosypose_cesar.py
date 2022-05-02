from cosypose.utils.tqdm import patch_tqdm; patch_tqdm()  # noqa
import torch.multiprocessing
import time
import json

from collections import OrderedDict
import yaml
import argparse

import torch
import numpy as np
import pandas as pd
import pickle as pkl
import logging

from cosypose.config import EXP_DIR, MEMORY, RESULTS_DIR, LOCAL_DATA_DIR

from cosypose.utils.distributed import init_distributed_mode, get_world_size

from cosypose.lib3d import Transform

from cosypose.lib3d.rigid_mesh_database import MeshDataBase
from cosypose.training.pose_models_cfg import create_model_refiner, create_model_coarse, check_update_config
from cosypose.rendering.bullet_batch_renderer import BulletBatchRenderer
from cosypose.integrated.pose_predictor import CoarseRefinePosePredictor
from cosypose.integrated.multiview_predictor import MultiviewScenePredictor

from cosypose.evaluation.meters.pose_meters import PoseErrorMeter
from cosypose.evaluation.pred_runner.multiview_predictions import MultiviewPredictionRunner
from cosypose.evaluation.pred_runner.bop_predictions import BopPredictionRunner
from cosypose.evaluation.eval_runner.pose_eval import PoseEvaluation

import cosypose.utils.tensor_collection as tc
from cosypose.evaluation.runner_utils import format_results, gather_predictions
from cosypose.utils.distributed import get_rank

# Detection
from cosypose.training.detector_models_cfg import create_model_detector
from cosypose.training.detector_models_cfg import check_update_config as check_update_config_detector
from cosypose.integrated.detector import Detector

from cosypose.datasets.datasets_cfg import make_scene_dataset, make_object_dataset
from cosypose.datasets.bop import remap_bop_targets
from cosypose.datasets.wrappers.multiview_wrapper import MultiViewWrapper

from cosypose.datasets.samplers import ListSampler
from cosypose.utils.logging import get_logger
logger = get_logger(__name__)

torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


@MEMORY.cache
def load_posecnn_results():
    results_path = LOCAL_DATA_DIR / 'saved_detections' / 'ycbv_posecnn.pkl'
    results = pkl.loads(results_path.read_bytes())
    infos, poses, bboxes = [], [], []

    l_offsets = (LOCAL_DATA_DIR / 'bop_datasets/ycbv' / 'offsets.txt').read_text().strip().split('\n')
    ycb_offsets = dict()
    for l_n in l_offsets:
        obj_id, offset = l_n[:2], l_n[3:]
        obj_id = int(obj_id)
        offset = np.array(json.loads(offset)) * 0.001
        ycb_offsets[obj_id] = offset

    def mat_from_qt(qt):
        wxyz = qt[:4].copy().tolist()
        xyzw = [*wxyz[1:], wxyz[0]]
        t = qt[4:].copy()
        return Transform(xyzw, t)

    for scene_view_str, result in results.items():
        scene_id, view_id = scene_view_str.split('/')
        scene_id, view_id = int(scene_id), int(view_id)
        n_dets = result['rois'].shape[0]
        for n in range(n_dets):
            obj_id = result['rois'][:, 1].astype(np.int)[n]
            label = f'obj_{obj_id:06d}'
            infos.append(dict(
                scene_id=scene_id,
                view_id=view_id,
                score=result['rois'][n, 1],
                label=label,
            ))
            bboxes.append(result['rois'][n, 2:6])
            pose = mat_from_qt(result['poses'][n])
            offset = ycb_offsets[obj_id]
            pose = pose * Transform((0, 0, 0, 1), offset).inverse()
            poses.append(pose.toHomogeneousMatrix())

    data = tc.PandasTensorCollection(
        infos=pd.DataFrame(infos),
        poses=torch.as_tensor(np.stack(poses)).float(),
        bboxes=torch.as_tensor(np.stack(bboxes)).float(),
    ).cpu()
    return data

def load_detector(run_id):
    run_dir = EXP_DIR / run_id
    cfg = yaml.load((run_dir / 'config.yaml').read_text(), Loader=yaml.FullLoader)
    cfg = check_update_config_detector(cfg)
    label_to_category_id = cfg.label_to_category_id
    model = create_model_detector(cfg, len(label_to_category_id))
    ckpt = torch.load(run_dir / 'checkpoint.pth.tar')
    ckpt = ckpt['state_dict']
    model.load_state_dict(ckpt)
    model = model.cuda().eval()
    model.cfg = cfg
    model.config = cfg
    model = Detector(model)
    return model

def load_models(coarse_run_id, refiner_run_id=None, n_workers=8, object_set='tless'):
    if object_set == 'tless':
        object_ds_name, urdf_ds_name = 'tless.bop', 'tless.cad'
    else:
        object_ds_name, urdf_ds_name = 'ycbv.bop-compat.eval', 'ycbv'

    object_ds = make_object_dataset(object_ds_name)
    mesh_db = MeshDataBase.from_object_ds(object_ds)
    renderer = BulletBatchRenderer(object_set=urdf_ds_name, n_workers=n_workers)
    mesh_db_batched = mesh_db.batched().cuda()

    def load_model(run_id):
        if run_id is None:
            return
        run_dir = EXP_DIR / run_id
        cfg = yaml.load((run_dir / 'config.yaml').read_text(), Loader=yaml.FullLoader)
        cfg = check_update_config(cfg)
        if cfg.train_refiner:
            model = create_model_refiner(cfg, renderer=renderer, mesh_db=mesh_db_batched)
            ckpt = torch.load(run_dir / 'checkpoint.pth.tar')
        else:
            model = create_model_coarse(cfg, renderer=renderer, mesh_db=mesh_db_batched)
            ckpt = torch.load(run_dir / 'checkpoint.pth.tar')
        ckpt = ckpt['state_dict']
        model.load_state_dict(ckpt)
        model = model.cuda().eval()
        model.cfg = cfg
        return model

    coarse_model = load_model(coarse_run_id)
    refiner_model = load_model(refiner_run_id)
    model = CoarseRefinePosePredictor(coarse_model=coarse_model,
                                      refiner_model=refiner_model)
    return model, mesh_db


def main():
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        if 'cosypose' in logger.name:
            logger.setLevel(logging.DEBUG)

    logger.info("Starting ...")
    init_distributed_mode()

    parser = argparse.ArgumentParser('Evaluation')
    parser.add_argument('--config', default='posecnn', type=str)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--job_dir', default='', type=str)
    parser.add_argument('--comment', default='', type=str)
    parser.add_argument('--nviews', dest='n_views', default=1, type=int)
    args = parser.parse_args()

    coarse_run_id = None
    refiner_run_id = None
    n_workers = 8
    n_plotters = 8
    n_views = 1

    n_frames = None
    scene_id = None
    group_id = None
    n_groups = None
    n_views = args.n_views
    skip_mv = args.n_views < 2
    skip_predictions = False

    object_set = 'ycbv'
    refiner_run_id = 'ycbv-refiner-finetune--251020'
    coarse_run_id = 'coarse-bop-ycbv-synt+real--822463'
    n_refiner_iterations = 2
    ds_name = 'ycbv.test.keyframes'

    n_rand = np.random.randint(1e10)
    save_dir = RESULTS_DIR / f'{args.config}-n_views={n_views}-{args.comment}-{n_rand}'
    logger.info(f"SAVE DIR: {save_dir}")
    logger.info(f"Coarse: {coarse_run_id}")
    logger.info(f"Refiner: {refiner_run_id}")

    # Load dataset
    scene_ds = make_scene_dataset(ds_name)

    if scene_id is not None:
        mask = scene_ds.frame_index['scene_id'] == scene_id
        scene_ds.frame_index = scene_ds.frame_index[mask].reset_index(drop=True)
    if n_frames is not None:
        scene_ds.frame_index = scene_ds.frame_index[mask].reset_index(drop=True)[:n_frames]

    # Predictions
    predictor, mesh_db = load_models(coarse_run_id, refiner_run_id, n_workers=n_plotters, object_set=object_set)
    mv_predictor = MultiviewScenePredictor(mesh_db)
    
    if 'posecnn' in args.config:
        n_coarse_iterations = 0
        prediction_label = 'posecnn'
        posecnn_detections = load_posecnn_results()
        pred_kwargs = {
                prediction_label : dict(
                    detections=posecnn_detections,
                    n_coarse_iterations=n_coarse_iterations,
                    n_refiner_iterations=n_refiner_iterations,
                    skip_mv=skip_mv,
                    pose_predictor=predictor,
                    mv_predictor=mv_predictor,
                    use_detections_TCO=posecnn_detections,
                ),
            }
    else:
        n_coarse_iterations = 1
        prediction_label = 'maskrcnn'
        detector_run_id = "detector-bop-ycbv-synt+real--292971"
        detector = load_detector(detector_run_id)
        pred_kwargs = {
                prediction_label : dict(
                    detector=detector,
                    pose_predictor=predictor,
                    n_coarse_iterations=n_coarse_iterations,
                    n_refiner_iterations=n_refiner_iterations,
                ),
            }
    scene_ds_pred = MultiViewWrapper(scene_ds, n_views=n_views)

    if 'posecnn' in args.config:
        pred_runner = MultiviewPredictionRunner(
            scene_ds_pred, batch_size=1, n_workers=n_workers,
            cache_data=len(pred_kwargs) > 1)
    else:
        pred_runner = BopPredictionRunner(scene_ds_pred, batch_size=1, n_workers=n_workers,
            cache_data=False)

    all_predictions = dict()
    for pred_prefix, pred_kwargs_n in pred_kwargs.items():
        logger.info(f"Prediction: {pred_prefix}")
        preds = pred_runner.get_predictions(**pred_kwargs_n)
        for preds_name, preds_n in preds.items():
            all_predictions[f'{pred_prefix}/{preds_name}'] = preds_n

    logger.info("Done with predictions")
    torch.distributed.barrier()

    all_predictions = OrderedDict({k: v for k, v in sorted(all_predictions.items(), key=lambda item: item[0])})
    
    # Saving results 
    best_pred =  all_predictions[f'{prediction_label}/refiner/iteration=2']
    views = np.zeros(len(best_pred))
    objs = np.zeros(len(best_pred))
    poses = np.zeros((len(best_pred),4,4))
    results = open("results.txt", "w")
    for i in range(len(best_pred)):
        marker = f"\n ----- detection n°{i} ---- \n"
        views[i] = best_pred[i].infos[1]
        if 'maskrcnn' in args.config:
            obj = best_pred[i].infos[2][4:]
            objs[i] = obj
        else:
            objs[i] = best_pred[i].infos[2]
        poses[i] = best_pred[i].poses.numpy()
        results.write(marker)
        results.write(f"View n°{views[i]} \n")
        results.write(f"Object n°{objs[i]} \n")
        results.write(np.array_str(poses[i]))
    results.close()
    np.savez('results.npz', view=views, obj=objs, pose=poses)



if __name__ == '__main__':
    patch_tqdm()
    main()
    time.sleep(2)
    if get_world_size() > 1:
        torch.distributed.barrier()
