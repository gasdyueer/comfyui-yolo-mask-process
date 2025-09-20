# ComfyUI YOLO Mask Process

一个强大的ComfyUI自定义节点项目，集成了YOLOv8目标检测和分割功能，并提供灵活的图像遮罩效果处理。

## 🚀 功能特性

### YOLO Detection and Segmentation 节点
- **多模型支持**：支持YOLOv8检测和分割模型
- **智能检测**：自动检测图像中的目标并生成边界框
- **分割支持**：支持实例分割，生成高质量的分割掩码
- **灵活过滤**：支持按类别ID或类别名称过滤特定目标
- **置信度控制**：可调节检测置信度阈值
- **掩码平滑**：可选的高斯平滑功能，提升掩码质量
- **详细输出**：提供带标注图像、合并掩码、独立掩码和检测结果JSON

### Masked Image Effects 节点
- **多样效果**：支持高斯模糊、马赛克等图像效果
- **灵活应用**：可应用于遮罩区域或非遮罩区域
- **参数可调**：所有效果参数均可精确控制
- **实时预览**：支持ComfyUI实时预览效果

## 📦 安装方法

### 1. 安装依赖

```bash
# 克隆项目到ComfyUI的custom_nodes目录
cd ComfyUI/custom_nodes
git clone https://github.com/gasdyueer/comfyui-yolo-mask-process.git

# 安装Python依赖
pip install ultralytics torch torchvision scipy pillow numpy
```

### 2. 下载YOLOv8模型

将YOLOv8模型文件放入 `ComfyUI/models/YOLO_MODEL/` 目录：

```bash
# 创建模型目录
mkdir -p ComfyUI/models/YOLO_MODEL

# 下载模型（例如YOLOv8s检测模型）
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt -O ComfyUI/models/YOLO_MODEL/yolov8s.pt

# 下载分割模型（可选）
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-seg.pt -O ComfyUI/models/YOLO_MODEL/yolov8s-seg.pt
```

### 3. 重启ComfyUI

安装完成后，重启ComfyUI以加载新节点。

## 🎯 使用指南

### YOLO Detection and Segmentation 节点

#### 输入参数：
- **image**: 输入图像（张量格式）
- **model_name**: YOLO模型名称（从YOLO_MODEL文件夹选择）
- **confidence_threshold**: 检测置信度阈值 (0.0-1.0)
- **filter_single_mask_by_class_id**: 按类别ID过滤 (0-999)
- **use_class_name_for_single_mask**: 启用类别名称过滤
- **class_name_for_single_mask**: 类别名称（当启用名称过滤时）
- **smooth_masks**: 启用掩码平滑
- **smooth_sigma**: 平滑强度 (0.0-10.0)

#### 输出：
- **PLOTTED_IMAGE**: 带检测框和标签的图像
- **SINGLE_COMBINED_MASK**: 单个合并掩码（可按类别过滤）
- **ALL_INDIVIDUAL_MASKS**: 所有独立实例掩码
- **DETECTIONS_JSON**: 包含所有检测结果的JSON数据

### Masked Image Effects 节点

#### 输入参数：
- **image**: 输入图像
- **mask**: 输入掩码
- **effect_type**: 效果类型
  - `gaussian_blur`: 高斯模糊
  - `mosaic`: 马赛克效果
- **apply_mode**: 应用模式
  - `masked_area`: 应用于遮罩白色区域
  - `unmasked_area`: 应用于遮罩黑色区域
- **mosaic_pixel_size**: 马赛克像素大小 (2-256)
- **gaussian_kernel_size**: 高斯核大小 (3-255，必须为奇数)
- **gaussian_sigma**: 高斯标准差 (0.1-20.0)

#### 输出：
- **IMAGE**: 处理后的图像

## 📋 模型要求

### 支持的模型格式：
- **检测模型**: `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, `yolov8x.pt`
- **分割模型**: `yolov8n-seg.pt`, `yolov8s-seg.pt`, `yolov8m-seg.pt`, `yolov8l-seg.pt`, `yolov8x-seg.pt`

### 模型放置位置：
```
ComfyUI/
├── models/
│   └── YOLO_MODEL/
│       ├── yolov8s.pt
│       ├── yolov8s-seg.pt
│       └── your_custom_model.pt
```

## 🛠️ 技术实现

### 核心功能：
1. **图像预处理**: 将ComfyUI张量转换为PIL图像格式
2. **模型推理**: 使用Ultralytics YOLOv8进行目标检测和分割
3. **结果处理**: 解析检测结果，生成掩码和标注图像
4. **效果应用**: 基于掩码对图像应用各种视觉效果
5. **输出生成**: 返回处理后的图像和相关数据

### 性能优化：
- GPU加速支持（当CUDA可用时）
- 内存高效的张量操作
- 可配置的置信度阈值
- 批量处理支持

## 📖 示例工作流

项目包含示例工作流文件，展示了如何将两个节点结合使用：

1. **YOLO检测**: 使用YOLOv8检测图像中的目标
2. **掩码处理**: 对检测到的目标区域应用图像效果
3. **结果预览**: 实时查看处理结果

### 快速开始：
1. 加载示例工作流：`example_workflows/detect and process.json`
2. 选择合适的YOLO模型
3. 调整检测参数
4. 选择图像效果类型
5. 运行工作流查看结果

## 🔧 故障排除

### 常见问题：

**1. 模型加载失败**
- 确保模型文件存在于 `ComfyUI/models/YOLO_MODEL/` 目录
- 检查模型文件是否损坏
- 确认模型文件名拼写正确

**2. CUDA错误**
- 确保PyTorch CUDA版本与系统CUDA版本兼容
- 检查GPU内存是否充足
- 尝试在CPU模式下运行（会较慢）

**3. 掩码尺寸不匹配**
- 节点会自动处理尺寸不匹配问题
- 检查输入图像和掩码的尺寸
- 如有需要，可以使用ComfyUI的Resize节点预处理

**4. 检测结果为空**
- 降低置信度阈值
- 检查模型是否支持当前图像中的目标类别
- 确认图像质量和光照条件

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

### 开发环境设置：
```bash
git clone https://github.com/gasdyueer/comfyui-yolo-mask-process.git
cd comfyui-yolo-mask-process
pip install -e .
```

## 📄 许可证

本项目采用MIT许可证。详见[LICENSE](LICENSE)文件。

## 🙏 致谢

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - 提供强大的目标检测和分割功能
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - 优秀的AI图像处理框架
- [PyTorch](https://pytorch.org/) - 深度学习框架

## 📞 支持

如有问题或建议，请：

1. 查看[Issues](https://github.com/gasdyueer/comfyui-yolo-mask-process/issues)页面
2. 创建新的Issue描述您的问题
3. 提供详细的错误信息和复现步骤

---

**版本**: 0.1.0
**兼容性**: ComfyUI 1.0+
**Python**: >=3.11