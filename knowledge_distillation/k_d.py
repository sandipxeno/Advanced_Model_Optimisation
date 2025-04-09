from ultralytics import YOLO
import torch
import torch.nn.functional as F


# --- Custom Knowledge Distillation Trainer ---
def build_kd_trainer_class(base_trainer_cls):
    class KDTrainer(base_trainer_cls):
        def __init__(self, overrides=None, _callbacks=None, teacher_path=None):
            self.teacher_path = teacher_path
            super().__init__(overrides=overrides, _callbacks=_callbacks)

        def setup_model(self):
            super().setup_model()

            # Load teacher model
            assert self.teacher_path is not None, "teacher_path must be set!"
            self.teacher = YOLO(self.teacher_path).model
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False

        def train_batch(self, batch):
            # Student forward pass
            student_out = self.model(batch['img'])
            loss, loss_items = self.criterion(student_out, batch)

            # Teacher forward pass (no grad)
            with torch.no_grad():
                teacher_out = self.teacher(batch['img'])

            # Distillation loss (L2)
            kd_loss = F.mse_loss(student_out[0], teacher_out[0])

            # Final loss
            total_loss = loss + 0.3 * kd_loss
            return total_loss, loss_items

    return KDTrainer


# --- Train with KD ---
if __name__ == "__main__":
    # Paths
    pruned_student_pth = r"C:\Users\mohas\Downloads\yolov12_quant_api-main\yolov12_quant_api-main\models\yolo12n_pruned.pt"
    teacher_pt = r"C:\Users\mohas\Downloads\yolov12_quant_api-main\yolov12_quant_api-main\models\yolo12n.pt"
    data_yaml = r"C:\Users\mohas\Downloads\yolov12_quant_api-main\yolov12_quant_api-main\coco128\coco128\coco128.yaml"
    output_dir = r"C:\Users\mohas\Downloads\yolov12_quant_api-main\yolov12_quant_api-main\models"

    # Load pruned student model
    student_model = YOLO(pruned_student_pth)

    # Init KD Trainer
    overrides = {
        'model': pruned_student_pth,
        'data': data_yaml,
        'epochs': 5,
        'imgsz': 640,
        'batch': 8,
        'project': output_dir,
        'name': 'distilled_pruned_model',
        'pretrained': False,
    }

    # Train with KD
    # Load model to determine base trainer
    dummy_model = YOLO(pruned_student_pth)
    BaseTrainerClass = dummy_model._smart_load("trainer")

    # Create KDTrainer subclass dynamically
    KDTrainer = build_kd_trainer_class(BaseTrainerClass)

    # Train with KD
    trainer = KDTrainer(overrides=overrides, teacher_path=teacher_pt)
    trainer.train()

