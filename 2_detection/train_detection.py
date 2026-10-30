
from ultralytics import YOLO
import torch


CONFIG = {
    "model":      "yolov8n.pt",     
    "epochs":     50,
    "imgsz":      384,             
    "batch":      16,              
    "lr0":        0.01,
    "patience":   10,              
    "project":    "runs/detection",
    "name":       "vnlp_yolov8",
    "device":     "0" if torch.cuda.is_available() else "cpu",
}


def train():
    print(f"🚀 Bắt đầu train Detection Model")
    print(f"   Device: {'GPU' if CONFIG['device'] != 'cpu' else 'CPU'}")
    print(f"   Model:  {CONFIG['model']}")
    print(f"   Epochs: {CONFIG['epochs']}")

    model = YOLO(CONFIG["model"])

    results = model.train(
        data      = CONFIG["data"],
        epochs    = CONFIG["epochs"],
        imgsz     = CONFIG["imgsz"],
        batch     = CONFIG["batch"],
        lr0       = CONFIG["lr0"],
        patience  = CONFIG["patience"],
        project   = CONFIG["project"],
        name      = CONFIG["name"],
        device    = CONFIG["device"],
        augment   = True,   
        cache     = True,  
        verbose   = True,
    )

    print(f"\n✅ Train xong!")
    print(f"   Best model: runs/detection/vnlp_yolov8/weights/best.pt")
    return results


def evaluate():
    """Đánh giá model trên tập test"""
    model = YOLO("runs/detection/vnlp_yolov8/weights/best.pt")
    metrics = model.val(
        data   = CONFIG["data"],
        split  = "test",
        imgsz  = CONFIG["imgsz"],
    )
    print(f"\n📊 Kết quả trên tập Test:")
    print(f"   mAP50:    {metrics.box.map50:.4f}")
    print(f"   mAP50-95: {metrics.box.map:.4f}")
    print(f"   Precision:{metrics.box.mp:.4f}")
    print(f"   Recall:   {metrics.box.mr:.4f}")
    return metrics


def detect_single_image(image_path: str, model_path: str = "runs/detection/vnlp_yolov8/weights/best.pt"):
    """
    Phát hiện biển số trong 1 ảnh
    Returns: list of {bbox: [x1,y1,x2,y2], confidence: float, type: str}
    """
    model = YOLO(model_path)
    results = model(image_path, imgsz=CONFIG["imgsz"])[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        w = x2 - x1
        h = y2 - y1
        plate_type = "one_row" if (w / h) > 3.0 else "two_row"
        detections.append({
            "bbox":       [int(x1), int(y1), int(x2), int(y2)],
            "confidence": round(conf, 4),
            "type":       plate_type,
        })

    return detections


if __name__ == "__main__":
    train()
    evaluate()