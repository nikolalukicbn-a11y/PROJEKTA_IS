#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SKRIPTA 4: PRIKAZ PO EPOHAMA
- Poredjenje Ground Truth vs Predikcija za svaku epohu
- Kreiranje GIF animacije napredovanja modela
"""

import sys
import glob as glob_mod
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image as PILImage
import random

from ultralytics import YOLO

sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# KONFIGURACIJA
# ============================================================
OUTPUT_DIR = Path('epoch_comparison')
OUTPUT_DIR.mkdir(exist_ok=True)
RANDOM_SEED = 42


def find_all_models():
    models = []

    runs_dir = Path('runs/detect')
    if runs_dir.exists():
        for subdir in runs_dir.iterdir():
            if not subdir.is_dir():
                continue
            weights_dir = subdir / 'weights'
            if weights_dir.exists():
                for pt in weights_dir.glob('*.pt'):
                    epoch_num = None
                    stem = pt.stem
                    if 'epoch' in stem:
                        try:
                            epoch_num = int(stem.replace('epoch', ''))
                        except ValueError:
                            epoch_num = 0
                    models.append((epoch_num, pt, subdir.name))

    models.sort(key=lambda x: x[0] if x[0] is not None else 999)
    return models


def find_best_model():
    runs_dir = Path('runs/detect')
    if runs_dir.exists():
        for subdir in sorted(runs_dir.iterdir(), reverse=True):
            if subdir.is_dir():
                best = subdir / 'weights' / 'best.pt'
                if best.exists():
                    return best
    for pt in Path('.').glob('best.pt'):
        return pt
    return None


def find_val_images():
    val_images = []
    val_annotations = {}

    possible = [
        ('val/images', 'val/labels'),
        ('images/val', 'labels/val'),
    ]

    for img_path, lbl_path in possible:
        img_dir = Path(img_path)
        lbl_dir = Path(lbl_path)
        if img_dir.exists() and lbl_dir.exists():
            for img_file in img_dir.glob('*'):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    ann_file = lbl_dir / f"{img_file.stem}.txt"
                    if ann_file.exists():
                        val_images.append(img_file)
                        val_annotations[img_file] = ann_file
            if val_images:
                break

    return val_images, val_annotations


def draw_ground_truth(ax, img_path, annotations_dict):
    ann_path = annotations_dict.get(img_path)
    if not ann_path or not ann_path.exists():
        return

    try:
        img = PILImage.open(img_path)
        img_w, img_h = img.size

        with open(ann_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, xc, yc, w, h = map(float, parts)

                x1 = (xc - w / 2) * img_w
                y1 = (yc - h / 2) * img_h
                x2 = (xc + w / 2) * img_w
                y2 = (yc + h / 2) * img_h

                boja = 'darkred' if cls == 0 else 'darkgreen'
                naziv = 'POSPAN' if cls == 0 else 'BUDAN'

                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                         linewidth=2, edgecolor=boja,
                                         facecolor='none', linestyle='--')
                ax.add_patch(rect)
                ax.text(x1, y1 - 3, f"GT: {naziv}",
                        fontsize=8, color='black', fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor=boja, alpha=0.5))
    except Exception:
        pass


def draw_predictions(ax, results):
    if len(results[0].boxes) == 0:
        ax.text(0.5, 0.5, "NEMA DETEKCIJE",
                transform=ax.transAxes, ha='center', va='center',
                fontsize=12, color='gray', fontweight='bold')
        return

    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()

    for box, conf, cls in zip(boxes, confs, classes):
        x1, y1, x2, y2 = box
        boja = 'red' if cls == 0 else 'lime'
        naziv = 'POSPAN' if cls == 0 else 'BUDAN'

        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=3, edgecolor=boja,
                                 facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f"{naziv}: {conf:.3f}",
                fontsize=9, color='white', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=boja, alpha=0.8))


def show_comparison(model, images, annotations, epoch_num, save_path=None):
    fig, axes = plt.subplots(len(images), 2, figsize=(14, 4 * len(images)))
    if len(images) == 1:
        axes = axes.reshape(1, -1)

    for idx, img_path in enumerate(images):
        ax_gt = axes[idx, 0]
        img = PILImage.open(img_path)
        ax_gt.imshow(img)
        ax_gt.axis('off')
        ax_gt.set_title("STVARNO STANJE (Ground Truth)", fontsize=10,
                        fontweight='bold', color='darkblue')
        draw_ground_truth(ax_gt, img_path, annotations)

        ax_pred = axes[idx, 1]
        ax_pred.imshow(img)
        ax_pred.axis('off')
        ax_pred.set_title(f"PREDIKCIJA — Epoha {epoch_num}", fontsize=10,
                          fontweight='bold', color='darkgreen')
        results = model(img_path)
        draw_predictions(ax_pred, results)

        for ax in [ax_gt, ax_pred]:
            ax.text(0.5, -0.08, img_path.name, transform=ax.transAxes,
                    ha='center', fontsize=8, color='gray')

    plt.suptitle(f'Epoha {epoch_num} — Ground Truth vs Predikcija',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return fig


def create_gif():
    image_files = sorted(
        glob_mod.glob(str(OUTPUT_DIR / 'epoha_*_comparison.png')),
        key=lambda x: int(x.split('epoha_')[1].split('_')[0])
    )

    if len(image_files) < 2:
        print("⚠️ Nema dovoljno slika za animaciju (potrebno 2+)")
        return

    images = [PILImage.open(f) for f in image_files]
    gif_path = OUTPUT_DIR / 'learning_progress.gif'
    images[0].save(gif_path, save_all=True, append_images=images[1:],
                   duration=800, loop=0)
    print(f"✅ GIF animacija: {gif_path}")


def main():
    print("=" * 60)
    print("SKRIPTA 4: PRIKAZ UČENJA PO EPOHAMA")
    print("=" * 60)

    models = find_all_models()
    valid_models = [(e, p, f) for e, p, f in models if e is not None]

    if not valid_models:
        best = find_best_model()
        if best:
            print("⚠️ Nema checkpoint modela, koristim samo finalni best.pt")
            valid_models = [('finalni', best, 'best_model')]
        else:
            print("❌ Nema pronađenih modela!")
            print("   Proveri da li je trening završen (02_trening.py)")
            sys.exit(1)

    print(f"📁 Pronađeno modela: {len(valid_models)}")
    for epoch, path, folder in valid_models[:5]:
        print(f"   epoha {epoch}: {path.name}")
    if len(valid_models) > 5:
        print(f"   ... i još {len(valid_models) - 5}")

    val_images, val_annotations = find_val_images()
    if not val_images:
        print("❌ Nema validacionih slika sa anotacijama!")
        val_dir = Path('val/images')
        if val_dir.exists():
            val_images = list(val_dir.glob('*.jpg')) + list(val_dir.glob('*.png'))
            print(f"   Koristim {len(val_images)} slika bez GT anotacija.")

    if not val_images:
        sys.exit(1)

    random.seed(RANDOM_SEED)
    sample_images = random.sample(val_images, min(4, len(val_images)))
    print(f"🖼️ Odabrano {len(sample_images)} slika za prikaz")

    print("\n🎯 Generisanje uporednih prikaza...")
    for epoch_num, model_path, folder_name in valid_models:
        print(f"   Epoha {epoch_num}...", end=' ')
        model = YOLO(model_path)
        save_path = OUTPUT_DIR / f'epoha_{epoch_num}_comparison.png'
        show_comparison(model, sample_images, val_annotations, epoch_num, save_path)
        print("✅")

    if len(valid_models) >= 2:
        print("\n🎬 Kreiranje GIF animacije...")
        create_gif()

    print("\n" + "=" * 60)
    print(f"✅ SKRIPTA 4 ZAVRŠENA — Rezultati u: {OUTPUT_DIR}")
    print(f"   epoha_X_comparison.png — uporedni prikaz za svaku epohu")
    if len(valid_models) >= 2:
        print(f"   learning_progress.gif  — animacija napredovanja")
    print("=" * 60)


if __name__ == "__main__":
    main()
