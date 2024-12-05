# Ultralytics YOLO 🚀, AGPL-3.0 license

from copy import copy

from ultralytics.models import yolo
from ultralytics.nn.tasks import CrowdingModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.plotting import plot_images, plot_results


class CrowdingTrainer(yolo.detect.DetectionTrainer):
    """
    A class extending the DetectionTrainer class for training based on a segmentation and pose model.

    Example:
        ```python
        from ultralytics.models.yolo.crowding import CrowdingTrainer

        args = dict(model='yolov8n-crowding.pt', data='coco8-crowding.yaml', epochs=3)
        trainer = CrowdingTrainer(overrides=args)
        trainer.train()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a CrowdingTrainer object with specified configurations and overrides."""
        if overrides is None:
            overrides = {}
        overrides["task"] = "crowding"
        super().__init__(cfg, overrides, _callbacks)

        if isinstance(self.args.device, str) and self.args.device.lower() == "mps":
            LOGGER.warning(
                "WARNING ⚠️ Apple MPS known Crowding bug. Recommend 'device=cpu' for Crowding models. "
                "See https://github.com/ultralytics/ultralytics/issues/4031."
            )

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Get crowding model with specified configuration and weights."""
        model = CrowdingModel(cfg, ch=3, nc=self.data["nc"], data_kpt_shape=self.data["kpt_shape"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        return model

    def set_model_attributes(self):
        """Sets keypoints shape attribute of PoseModel."""
        super().set_model_attributes()
        self.model.kpt_shape = self.data["kpt_shape"]

    def get_validator(self):
        """Returns an instance of the PoseValidator class for validation."""
        self.loss_names = "box_loss", "seg_loss", "pose_loss", "kobj_loss", "size_loss", "cls_loss", "dfl_loss"
        return yolo.crowding.CrowdingValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def plot_training_samples(self, batch, ni):
        """Plot a batch of training samples with annotated class labels, bounding boxes, and keypoints."""
        images = batch["img"]
        kpts = batch["keypoints"]
        cls = batch["cls"].squeeze(-1)
        bboxes = batch["bboxes"]
        paths = batch["im_file"]
        batch_idx = batch["batch_idx"]
        plot_images(
            images,
            batch_idx,
            cls,
            bboxes,
            masks=batch["masks"],
            kpts=kpts,
            paths=paths,
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_metrics(self):
        """Plots training/val metrics."""
        plot_results(file=self.csv, segment=True, pose=True, on_plot=self.on_plot)  # save results.png
