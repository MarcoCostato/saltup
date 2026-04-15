import numpy as np


def load_yolo_output(filename, grid_h=15, grid_w=20, num_anchors=5, num_classes=4):
    """
    Carica l'output YOLOv2 da un file binario e lo riorganizza nella forma corretta.
    """
    anchor_stride = num_classes + 5  # Per ogni anchor: (x, y, w, h, obj_score + class scores)
    raw_output = np.fromfile(filename, dtype=np.float32)

    return raw_output.reshape((grid_h, grid_w, num_anchors, anchor_stride)), raw_output

def process_yolo_output(raw_output, flat_output, anchors, conf_threshold=0.5):
    """
    Processa l'output YOLOv2 e stampa gli offset C-style delle bbox trovate.
    """
    grid_h, grid_w, num_anchors = raw_output.shape[:3]
    boxes = []

    for row in range(grid_h):
        for col in range(grid_w):
            for a in range(num_anchors):
                offset = ((row * grid_w + col) * num_anchors + a) * 9  # Offset in array
                pred = raw_output[row, col, a]

                # Estrai valori e attiva funzioni
                tx, ty, tw, th, obj_score = pred[:5]
                obj_score = 1 / (1 + np.exp(-obj_score))  # Sigmoid
                class_scores = np.exp(pred[5:])
                class_probs = class_scores / np.sum(class_scores) * obj_score

                best_class = np.argmax(class_probs)
                best_score = class_probs[best_class]

                if best_score > conf_threshold:
                    bx = (col + 1 / (1 + np.exp(-tx))) / grid_w
                    by = (row + 1 / (1 + np.exp(-ty))) / grid_h
                    bw = anchors[2 * a] * np.exp(tw)
                    bh = anchors[2 * a + 1] * np.exp(th)

                    boxes.append((bx, by, bw, bh, best_score, best_class, offset))

    # Stampa come in C gli offset delle bbox trovate
    print("\n=== Offsets delle BBox trovate ===")
    for x, y, w, h, conf, cls, offset in boxes:
        print(f"Offset: {offset}, Class: {cls}, Conf: {conf:.2f}, BBox: ({x:.3f}, {y:.3f}, {w:.3f}, {h:.3f})")

    return boxes


# === ESEMPIO DI UTILIZZO ===
if __name__ == "__main__":
    # filename = "golo_python_dump.bin"
    filename = "/home/mgarzola/GolandProjects/golo/rawOutput.bin"

    anchors = [
        0.14, 0.19,   # Anchor box 1
        0.13, 0.52,   # Anchor box 2
        0.16, 0.31,   # Anchor box 3
        0.45, 0.62,   # Anchor box 4
        0.28, 0.38    # Anchor box 5
    ]

    raw_output, flat_output = load_yolo_output(filename)
    process_yolo_output(raw_output, flat_output, anchors)
