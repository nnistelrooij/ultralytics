# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, LOGGER, ops


class CrowdingPredictor(DetectionPredictor):
    """
    A class extending the DetectionPredictor class for prediction based on a segmentation and pose model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.crowding import CrowdingPredictor

        args = dict(model='yolov8n-crowding.pt', source=ASSETS)
        predictor = CrowdingPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initializes CrowdingPredictor, sets task to 'crowding' and logs a warning for using 'mps' as device."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "crowding"
        if isinstance(self.args.device, str) and self.args.device.lower() == "mps":
            LOGGER.warning(
                "WARNING ⚠️ Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. "
                "See https://github.com/ultralytics/ultralytics/issues/4031."
            )

    def postprocess(self, preds, img, orig_imgs):
        """Return detection results for a given input image or list of images."""
        p = ops.non_max_suppression(
            preds[0],
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
            nc=len(self.model.names),
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        proto = preds[1][-3] if isinstance(preds[1], tuple) else preds[1]  # tuple if PyTorch model or array if exported
        for i, pred in enumerate(p):
            orig_img = orig_imgs[i]
            img_path = self.batch[0][i]

            nk = 6
            if not len(pred):  # save empty boxes
                masks = None
            elif self.args.retina_masks:
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                masks = ops.process_mask_native(proto[i], pred[:, 6:-4 - nk], pred[:, :4], orig_img.shape[:2])  # HWC
            else:
                masks = ops.process_mask(proto[i], pred[:, 6:-2 - nk], pred[:, :4], img.shape[2:], upsample=True)  # HWC
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)


            pred_kpts = pred[:, -2 - nk:-2].view(len(pred), -1, 3) if len(pred) else pred[:, -2 - nk:-2]
            pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts.clone(), orig_img.shape)

            results.append(Results(
                orig_img, path=img_path, names=self.model.names,
                boxes=pred[:, :6], masks=masks, keypoints=pred_kpts, sizes=pred[:, -2:],
            ))
        return results
