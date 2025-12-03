#!/usr/bin/env python3
"""
YOLO数据集分析脚本
分析数据集的数量、质量、分布、标注完整性等信息
支持: 关键点检测(pose)和实例分割(seg)数据集

使用方法:
    python analyze_dataset.py --data_dir ../pose_data --task pose --output_dir ./
    python analyze_dataset.py --data_dir ../seg_data --task seg --output_dir ./
"""

import os
import argparse
import yaml
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# 设置字体
plt.rcParams['axes.unicode_minus'] = False


class DatasetAnalyzer:
    """YOLO数据集分析器"""

    def __init__(self, data_dir: str, task: str = 'pose', output_dir: str = './'):
        """
        Args:
            data_dir: 数据集根目录 (包含images和labels文件夹)
            task: 任务类型 'pose' 或 'seg'
            output_dir: 输出目录
        """
        self.data_dir = Path(data_dir)
        self.task = task
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 数据集配置
        self.yaml_path = self.data_dir / 'dataset.yaml'
        self.config = self._load_config()

        # 统计数据
        self.stats = {
            'summary': {},
            'images': [],
            'labels': [],
            'issues': []
        }

    def _load_config(self) -> dict:
        """加载dataset.yaml配置"""
        if self.yaml_path.exists():
            with open(self.yaml_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {}

    def analyze(self):
        """执行完整分析"""
        print("=" * 60)
        print(f"YOLO数据集分析 - {self.task.upper()}")
        print(f"数据目录: {self.data_dir}")
        print("=" * 60)

        # 1. 分析各个split
        for split in ['train', 'val', 'test']:
            self._analyze_split(split)

        # 2. 生成汇总统计
        self._generate_summary()

        # 3. 生成可视化
        self._generate_visualizations()

        # 4. 保存报告
        self._save_report()

        print("\n✅ 分析完成!")
        print(f"报告保存至: {self.output_dir}")

    def _analyze_split(self, split: str):
        """分析单个数据集split"""
        images_dir = self.data_dir / 'images' / split
        labels_dir = self.data_dir / 'labels' / split

        if not images_dir.exists():
            print(f"⚠️  {split} 目录不存在，跳过")
            return

        print(f"\n📂 分析 {split} 集...")

        # 获取所有图像文件
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [f for f in images_dir.iterdir()
                       if f.suffix.lower() in image_extensions]

        split_stats = {
            'split': split,
            'num_images': len(image_files),
            'num_labels': 0,
            'missing_labels': [],
            'empty_labels': [],
            'image_sizes': [],
            'objects_per_image': [],
            'class_distribution': defaultdict(int),
            'bbox_sizes': [],
            'keypoints_stats': [] if self.task == 'pose' else None,
            'polygon_stats': [] if self.task == 'seg' else None,
        }

        for img_path in tqdm(image_files, desc=f"  处理 {split}"):
            self._analyze_image(img_path, labels_dir, split_stats)

        self.stats['images'].append(split_stats)

        # 打印split统计
        print(f"  图像数量: {split_stats['num_images']}")
        print(f"  标签数量: {split_stats['num_labels']}")
        print(f"  缺失标签: {len(split_stats['missing_labels'])}")
        print(f"  空标签: {len(split_stats['empty_labels'])}")

    def _analyze_image(self, img_path: Path, labels_dir: Path, split_stats: dict):
        """分析单张图像及其标签"""
        # 读取图像尺寸
        try:
            with Image.open(img_path) as img:
                width, height = img.size
                split_stats['image_sizes'].append({
                    'file': img_path.name,
                    'width': width,
                    'height': height,
                    'aspect_ratio': width / height if height > 0 else 0
                })
        except Exception as e:
            self.stats['issues'].append({
                'type': 'image_read_error',
                'file': str(img_path),
                'error': str(e)
            })
            return

        # 查找对应标签文件
        label_path = labels_dir / (img_path.stem + '.txt')

        if not label_path.exists():
            split_stats['missing_labels'].append(img_path.name)
            split_stats['objects_per_image'].append(0)
            return

        split_stats['num_labels'] += 1

        # 读取标签
        with open(label_path, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        if not lines:
            split_stats['empty_labels'].append(img_path.name)
            split_stats['objects_per_image'].append(0)
            return

        split_stats['objects_per_image'].append(len(lines))

        # 解析每个目标
        for line in lines:
            self._parse_label_line(line, width, height, split_stats)

    def _parse_label_line(self, line: str, img_w: int, img_h: int, split_stats: dict):
        """解析单行标签"""
        parts = line.split()
        if len(parts) < 5:
            return

        class_id = int(parts[0])
        split_stats['class_distribution'][class_id] += 1

        if self.task == 'pose':
            # YOLO pose格式: class x_center y_center width height kp1_x kp1_y kp1_v ...
            if len(parts) >= 5:
                x_c, y_c, w, h = map(float, parts[1:5])
                split_stats['bbox_sizes'].append({
                    'width': w * img_w,
                    'height': h * img_h,
                    'area': w * h * img_w * img_h,
                    'aspect_ratio': w / h if h > 0 else 0
                })

                # 解析关键点
                kpts = parts[5:]
                if len(kpts) >= 3:
                    num_kpts = len(kpts) // 3
                    kpt_data = {'num_keypoints': num_kpts, 'visible': 0, 'invisible': 0, 'missing': 0}
                    for i in range(num_kpts):
                        try:
                            v = int(float(kpts[i * 3 + 2]))
                            if v == 2:
                                kpt_data['visible'] += 1
                            elif v == 1:
                                kpt_data['invisible'] += 1
                            else:
                                kpt_data['missing'] += 1
                        except (IndexError, ValueError):
                            kpt_data['missing'] += 1
                    split_stats['keypoints_stats'].append(kpt_data)

        elif self.task == 'seg':
            # YOLO seg格式: class x1 y1 x2 y2 ... (多边形坐标)
            # 前5个值可能包含bbox信息，之后是多边形点
            coords = list(map(float, parts[1:]))
            num_points = len(coords) // 2

            if num_points > 0:
                # 计算多边形面积（使用Shoelace公式）
                xs = [coords[i * 2] * img_w for i in range(num_points)]
                ys = [coords[i * 2 + 1] * img_h for i in range(num_points)]

                # Shoelace公式计算面积
                area = 0.5 * abs(sum(xs[i] * ys[(i + 1) % num_points] -
                                     xs[(i + 1) % num_points] * ys[i]
                                     for i in range(num_points)))

                split_stats['polygon_stats'].append({
                    'num_points': num_points,
                    'area': area,
                    'bbox_w': max(xs) - min(xs) if xs else 0,
                    'bbox_h': max(ys) - min(ys) if ys else 0
                })

                # 估算bbox
                if xs and ys:
                    w = max(xs) - min(xs)
                    h = max(ys) - min(ys)
                    split_stats['bbox_sizes'].append({
                        'width': w,
                        'height': h,
                        'area': area,
                        'aspect_ratio': w / h if h > 0 else 0
                    })

    def _generate_summary(self):
        """生成汇总统计"""
        total_images = sum(s['num_images'] for s in self.stats['images'])
        total_labels = sum(s['num_labels'] for s in self.stats['images'])
        total_missing = sum(len(s['missing_labels']) for s in self.stats['images'])
        total_empty = sum(len(s['empty_labels']) for s in self.stats['images'])

        # 合并所有类别分布
        all_classes = defaultdict(int)
        all_bbox_sizes = []
        all_objects = []
        all_img_sizes = []

        for s in self.stats['images']:
            for cls_id, count in s['class_distribution'].items():
                all_classes[cls_id] += count
            all_bbox_sizes.extend(s['bbox_sizes'])
            all_objects.extend(s['objects_per_image'])
            all_img_sizes.extend(s['image_sizes'])

        # 图像尺寸统计
        if all_img_sizes:
            widths = [s['width'] for s in all_img_sizes]
            heights = [s['height'] for s in all_img_sizes]
            img_size_stats = {
                'width_min': min(widths),
                'width_max': max(widths),
                'width_mean': np.mean(widths),
                'height_min': min(heights),
                'height_max': max(heights),
                'height_mean': np.mean(heights),
            }
        else:
            img_size_stats = {}

        # 目标尺寸统计
        if all_bbox_sizes:
            areas = [b['area'] for b in all_bbox_sizes]
            bbox_stats = {
                'area_min': min(areas),
                'area_max': max(areas),
                'area_mean': np.mean(areas),
                'area_median': np.median(areas),
            }
        else:
            bbox_stats = {}

        self.stats['summary'] = {
            'task': self.task,
            'data_dir': str(self.data_dir),
            'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_images': total_images,
            'total_labels': total_labels,
            'total_objects': sum(all_objects),
            'missing_labels': total_missing,
            'empty_labels': total_empty,
            'label_completeness': f"{(total_labels / total_images * 100):.2f}%" if total_images > 0 else "N/A",
            'avg_objects_per_image': np.mean(all_objects) if all_objects else 0,
            'num_classes': len(all_classes),
            'class_distribution': dict(all_classes),
            'image_size_stats': img_size_stats,
            'bbox_size_stats': bbox_stats,
            'splits': {s['split']: s['num_images'] for s in self.stats['images']},
            'issues': self.stats['issues']
        }

        # 关键点统计
        if self.task == 'pose':
            all_kpt_stats = []
            for s in self.stats['images']:
                if s['keypoints_stats']:
                    all_kpt_stats.extend(s['keypoints_stats'])

            if all_kpt_stats:
                total_visible = sum(k['visible'] for k in all_kpt_stats)
                total_invisible = sum(k['invisible'] for k in all_kpt_stats)
                total_missing = sum(k['missing'] for k in all_kpt_stats)
                total_kpts = total_visible + total_invisible + total_missing

                self.stats['summary']['keypoint_stats'] = {
                    'total_keypoints': total_kpts,
                    'visible_rate': f"{(total_visible / total_kpts * 100):.2f}%" if total_kpts > 0 else "N/A",
                    'invisible_rate': f"{(total_invisible / total_kpts * 100):.2f}%" if total_kpts > 0 else "N/A",
                    'missing_rate': f"{(total_missing / total_kpts * 100):.2f}%" if total_kpts > 0 else "N/A",
                }

        # 分割统计
        if self.task == 'seg':
            all_poly_stats = []
            for s in self.stats['images']:
                if s['polygon_stats']:
                    all_poly_stats.extend(s['polygon_stats'])

            if all_poly_stats:
                num_points = [p['num_points'] for p in all_poly_stats]
                self.stats['summary']['polygon_stats'] = {
                    'avg_points_per_mask': np.mean(num_points),
                    'min_points': min(num_points),
                    'max_points': max(num_points),
                }


    def _generate_visualizations(self):
        """生成可视化图表"""
        print("\n📊 生成可视化图表...")

        # 收集所有数据
        all_objects = []
        all_bbox_sizes = []
        all_img_sizes = []
        all_classes = defaultdict(int)
        split_counts = {}

        for s in self.stats['images']:
            split_counts[s['split']] = s['num_images']
            all_objects.extend(s['objects_per_image'])
            all_bbox_sizes.extend(s['bbox_sizes'])
            all_img_sizes.extend(s['image_sizes'])
            for cls_id, count in s['class_distribution'].items():
                all_classes[cls_id] += count

        # 创建图表
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Dataset Analysis Report - {self.task.upper()}', fontsize=16, fontweight='bold')

        # 1. 数据集划分饼图
        ax1 = axes[0, 0]
        if split_counts:
            labels = list(split_counts.keys())
            sizes = list(split_counts.values())
            colors = ['#66b3ff', '#99ff99', '#ffcc99'][:len(labels)]
            ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
            ax1.set_title('Dataset Split')

        # 2. 每张图像目标数量分布
        ax2 = axes[0, 1]
        if all_objects:
            ax2.hist(all_objects, bins=max(10, max(all_objects) - min(all_objects) + 1),
                     color='steelblue', edgecolor='black', alpha=0.7)
            ax2.axvline(np.mean(all_objects), color='red', linestyle='--',
                       label=f'Mean: {np.mean(all_objects):.2f}')
            ax2.set_xlabel('Objects per Image')
            ax2.set_ylabel('Number of Images')
            ax2.set_title('Objects Distribution')
            ax2.legend()

        # 3. 类别分布
        ax3 = axes[0, 2]
        if all_classes:
            class_names = self.config.get('names', [f'class_{i}' for i in all_classes.keys()])
            class_ids = sorted(all_classes.keys())
            counts = [all_classes[i] for i in class_ids]
            names = [class_names[i] if i < len(class_names) else f'class_{i}' for i in class_ids]

            bars = ax3.bar(range(len(class_ids)), counts, color='coral', edgecolor='black')
            ax3.set_xticks(range(len(class_ids)))
            ax3.set_xticklabels(names, rotation=45, ha='right')
            ax3.set_xlabel('Class')
            ax3.set_ylabel('Count')
            ax3.set_title('Class Distribution')

            # 添加数值标签
            for bar, count in zip(bars, counts):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        str(count), ha='center', va='bottom', fontsize=8)

        # 4. 图像尺寸分布
        ax4 = axes[1, 0]
        if all_img_sizes:
            widths = [s['width'] for s in all_img_sizes]
            heights = [s['height'] for s in all_img_sizes]
            ax4.scatter(widths, heights, alpha=0.6, c='green', edgecolors='darkgreen')
            ax4.set_xlabel('Image Width (pixels)')
            ax4.set_ylabel('Image Height (pixels)')
            ax4.set_title('Image Size Distribution')
            ax4.grid(True, alpha=0.3)

        # 5. 目标尺寸分布 (宽高比)
        ax5 = axes[1, 1]
        if all_bbox_sizes:
            aspect_ratios = [b['aspect_ratio'] for b in all_bbox_sizes if b['aspect_ratio'] > 0]
            if aspect_ratios:
                ax5.hist(aspect_ratios, bins=30, color='purple', edgecolor='black', alpha=0.7)
                ax5.axvline(np.median(aspect_ratios), color='red', linestyle='--',
                           label=f'Median: {np.median(aspect_ratios):.2f}')
                ax5.set_xlabel('Aspect Ratio (W/H)')
                ax5.set_ylabel('Count')
                ax5.set_title('Bbox Aspect Ratio Distribution')
                ax5.legend()

        # 6. 目标面积分布
        ax6 = axes[1, 2]
        if all_bbox_sizes:
            areas = [b['area'] for b in all_bbox_sizes]
            # 使用对数刻度更好地显示分布
            areas_log = [np.log10(a) if a > 0 else 0 for a in areas]
            ax6.hist(areas_log, bins=30, color='orange', edgecolor='black', alpha=0.7)
            ax6.set_xlabel('Object Area (log10 pixels^2)')
            ax6.set_ylabel('Count')
            ax6.set_title('Object Area Distribution')

        plt.tight_layout()

        # 保存图表
        fig_path = self.output_dir / f'{self.task}_dataset_analysis.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  保存图表: {fig_path}")

        # 如果是pose任务，生成关键点可见性图表
        if self.task == 'pose':
            self._plot_keypoint_visibility()

    def _plot_keypoint_visibility(self):
        """生成关键点可见性图表"""
        all_kpt_stats = []
        for s in self.stats['images']:
            if s['keypoints_stats']:
                all_kpt_stats.extend(s['keypoints_stats'])

        if not all_kpt_stats:
            return

        # 计算每个关键点的可见性
        num_kpts = all_kpt_stats[0]['num_keypoints'] if all_kpt_stats else 0
        kpt_names = self.config.get('kpt_names', [f'kpt_{i}' for i in range(num_kpts)])

        total_visible = sum(k['visible'] for k in all_kpt_stats)
        total_invisible = sum(k['invisible'] for k in all_kpt_stats)
        total_missing = sum(k['missing'] for k in all_kpt_stats)

        fig, ax = plt.subplots(figsize=(8, 6))
        labels = ['Visible (v=2)', 'Occluded (v=1)', 'Missing (v=0)']
        sizes = [total_visible, total_invisible, total_missing]
        colors = ['#2ecc71', '#f39c12', '#e74c3c']

        ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title('Keypoint Visibility Distribution')

        fig_path = self.output_dir / f'{self.task}_keypoint_visibility.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  保存图表: {fig_path}")

    def _save_report(self):
        """保存分析报告"""
        print("\n💾 保存报告...")

        # 保存JSON报告
        json_path = self.output_dir / f'{self.task}_analysis_report.json'

        # 将numpy类型转换为Python原生类型
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        report_data = convert_types(self.stats['summary'])

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        print(f"  保存JSON报告: {json_path}")

        # 保存文本报告
        txt_path = self.output_dir / f'{self.task}_analysis_report.txt'
        self._write_text_report(txt_path)
        print(f"  保存文本报告: {txt_path}")

        # 保存问题列表
        if self.stats['issues']:
            issues_path = self.output_dir / f'{self.task}_issues.json'
            with open(issues_path, 'w', encoding='utf-8') as f:
                json.dump(self.stats['issues'], f, ensure_ascii=False, indent=2)
            print(f"  保存问题列表: {issues_path}")

        # 保存缺失/空标签列表
        missing_labels = []
        empty_labels = []
        for s in self.stats['images']:
            missing_labels.extend([(s['split'], f) for f in s['missing_labels']])
            empty_labels.extend([(s['split'], f) for f in s['empty_labels']])

        if missing_labels:
            missing_path = self.output_dir / f'{self.task}_missing_labels.txt'
            with open(missing_path, 'w') as f:
                f.write("# 缺失标签的图像列表\n")
                f.write("# 格式: split, image_file\n\n")
                for split, img in missing_labels:
                    f.write(f"{split}, {img}\n")
            print(f"  保存缺失标签列表: {missing_path}")

        if empty_labels:
            empty_path = self.output_dir / f'{self.task}_empty_labels.txt'
            with open(empty_path, 'w') as f:
                f.write("# 空标签的图像列表\n")
                f.write("# 格式: split, image_file\n\n")
                for split, img in empty_labels:
                    f.write(f"{split}, {img}\n")
            print(f"  保存空标签列表: {empty_path}")

    def _write_text_report(self, path: Path):
        """写入文本格式报告"""
        summary = self.stats['summary']

        lines = [
            "=" * 70,
            f"YOLO 数据集分析报告 - {summary['task'].upper()}",
            "=" * 70,
            f"分析时间: {summary['analysis_time']}",
            f"数据目录: {summary['data_dir']}",
            "",
            "-" * 70,
            "📊 基本统计",
            "-" * 70,
            f"总图像数量:     {summary['total_images']}",
            f"总标签数量:     {summary['total_labels']}",
            f"总目标数量:     {summary['total_objects']}",
            f"标签完整性:     {summary['label_completeness']}",
            f"缺失标签数:     {summary['missing_labels']}",
            f"空标签数:       {summary['empty_labels']}",
            f"平均目标/图像:  {summary['avg_objects_per_image']:.2f}",
            f"类别数量:       {summary['num_classes']}",
            "",
            "-" * 70,
            "📁 数据集划分",
            "-" * 70,
        ]

        for split, count in summary['splits'].items():
            ratio = count / summary['total_images'] * 100 if summary['total_images'] > 0 else 0
            lines.append(f"  {split:10s}: {count:6d} ({ratio:.1f}%)")

        lines.extend([
            "",
            "-" * 70,
            "📏 图像尺寸统计",
            "-" * 70,
        ])

        if summary['image_size_stats']:
            iss = summary['image_size_stats']
            lines.extend([
                f"  宽度范围: {iss['width_min']:.0f} - {iss['width_max']:.0f} (平均: {iss['width_mean']:.0f})",
                f"  高度范围: {iss['height_min']:.0f} - {iss['height_max']:.0f} (平均: {iss['height_mean']:.0f})",
            ])

        lines.extend([
            "",
            "-" * 70,
            "🎯 目标尺寸统计",
            "-" * 70,
        ])

        if summary['bbox_size_stats']:
            bss = summary['bbox_size_stats']
            lines.extend([
                f"  面积范围: {bss['area_min']:.0f} - {bss['area_max']:.0f} pixels²",
                f"  面积平均: {bss['area_mean']:.0f} pixels²",
                f"  面积中位数: {bss['area_median']:.0f} pixels²",
            ])

        lines.extend([
            "",
            "-" * 70,
            "🏷️ 类别分布",
            "-" * 70,
        ])

        class_names = self.config.get('names', [])
        for cls_id, count in sorted(summary['class_distribution'].items()):
            name = class_names[cls_id] if cls_id < len(class_names) else f'class_{cls_id}'
            lines.append(f"  {cls_id}: {name:20s} - {count:6d}")

        # 关键点统计
        if 'keypoint_stats' in summary:
            kps = summary['keypoint_stats']
            lines.extend([
                "",
                "-" * 70,
                "🦴 关键点统计",
                "-" * 70,
                f"  总关键点数:   {kps['total_keypoints']}",
                f"  可见率:       {kps['visible_rate']}",
                f"  不可见率:     {kps['invisible_rate']}",
                f"  缺失率:       {kps['missing_rate']}",
            ])

        # 分割统计
        if 'polygon_stats' in summary:
            ps = summary['polygon_stats']
            lines.extend([
                "",
                "-" * 70,
                "🎭 分割掩码统计",
                "-" * 70,
                f"  平均点数/掩码: {ps['avg_points_per_mask']:.1f}",
                f"  最少点数:      {ps['min_points']}",
                f"  最多点数:      {ps['max_points']}",
            ])

        lines.extend([
            "",
            "=" * 70,
            "报告生成完成",
            "=" * 70,
        ])

        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description='YOLO数据集分析工具')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='数据集根目录 (包含images和labels文件夹)')
    parser.add_argument('--task', type=str, choices=['pose', 'seg'], default='pose',
                        help='任务类型: pose(关键点检测) 或 seg(实例分割)')
    parser.add_argument('--output_dir', type=str, default='./',
                        help='输出目录 (默认当前目录)')

    args = parser.parse_args()

    analyzer = DatasetAnalyzer(
        data_dir=args.data_dir,
        task=args.task,
        output_dir=args.output_dir
    )
    analyzer.analyze()


if __name__ == '__main__':
    main()
