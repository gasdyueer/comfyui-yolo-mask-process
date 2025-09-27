# ComfyUI YOLO Mask Process

一个强大的ComfyUI自定义节点项目，集成了YOLOv8目标检测和分割功能，并提供灵活的图像遮罩效果处理。

## 🚀 功能特性

### YOLO Detection and Segmentation 节点
- **多模型支持**：支持YOLOv8检测和分割模型（yolov8n.pt, yolov8s.pt等）
- **智能检测**：自动检测图像中的目标并生成边界框和置信度分数
- **分割支持**：支持实例分割，生成高质量的分割掩码
- **灵活过滤**：支持按类别ID或类别名称过滤特定目标（不区分大小写）
- **置信度控制**：可调节检测置信度阈值 (0.0-1.0)
- **掩码平滑**：可选的高斯平滑功能，提升掩码边缘质量（0.0-10.0强度控制）
- **强制掩码输出**：当分割模型无掩码时，可使用边界框自动生成掩码
- **自动尺寸适配**：自动处理模型输出与输入图像尺寸不匹配的问题
- **多输出模式**：
  - 带标注图像（边界框和类别标签）
  - 单个合并掩码（可按类别过滤）
  - 所有独立实例掩码
  - 详细检测结果JSON（包含类别、置信度、边界框等元数据）

### Masked Image Effects 节点
- **多样效果**：支持高斯模糊、马赛克等图像特效
- **灵活应用区域**：可应用于遮罩白色区域(masked_area)或黑色区域(unmasked_area)
- **智能尺寸适配**：自动处理输入掩码与图像尺寸不匹配的情况
- **参数精确控制**：
  - 马赛克：像素块大小控制(2-256)
  - 高斯模糊：核大小(3-255，必须奇数)和标准差(0.1-20.0)
- **多后端支持**：优先使用torchvision，自动降级到PIL实现
- **实时预览**：支持ComfyUI实时预览处理效果
- **内存优化**：高效的张量操作，减少内存占用

## 📦 安装方法

### 1. 安装依赖

```bash
# 克隆项目到ComfyUI的custom_nodes目录
cd ComfyUI/custom_nodes
git clone https://github.com/gasdyueer/comfyui-yolo-mask-process.git

# 进入项目目录
cd comfyui-yolo-mask-process

# 安装Python依赖（推荐使用requirements.txt）
pip install -r requirements.txt

# 或者手动安装核心依赖
pip install ultralytics>=8.2.0 torch torchvision scipy pillow numpy

# 可选：安装开发依赖
pip install -e ".[dev]"
```

### 依赖说明：
- **ultralytics**: YOLOv8模型推理引擎
- **torch & torchvision**: 深度学习框架和图像处理
- **scipy**: 科学计算，用于掩码平滑
- **pillow**: 图像处理基础库
- **numpy**: 数值计算支持

### 2. 下载YOLOv8模型

将YOLOv8模型文件放入 `ComfyUI/models/YOLO_MODEL/` 目录：

```bash
# 创建模型目录
mkdir -p ComfyUI/models/YOLO_MODEL

# 下载模型（选择适合您需求的大小和类型）
# 检测模型 - 用于目标检测和边界框生成
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt -O ComfyUI/models/YOLO_MODEL/yolov8n.pt
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt -O ComfyUI/models/YOLO_MODEL/yolov8s.pt
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt -O ComfyUI/models/YOLO_MODEL/yolov8m.pt

# 分割模型 - 用于实例分割和掩码生成
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-seg.pt -O ComfyUI/models/YOLO_MODEL/yolov8n-seg.pt
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-seg.pt -O ComfyUI/models/YOLO_MODEL/yolov8s-seg.pt
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m-seg.pt -O ComfyUI/models/YOLO_MODEL/yolov8m-seg.pt
```

### 模型选择建议：
- **yolov8n(-seg).pt**: Nano模型，最快，精度适中，适合实时应用
- **yolov8s(-seg).pt**: Small模型，平衡速度和精度，推荐用于大多数应用
- **yolov8m(-seg).pt**: Medium模型，更高精度，适合对准确性要求较高的场景

### 3. 重启ComfyUI

安装完成后，重启ComfyUI以加载新节点。

## 🎯 使用指南

### YOLO Detection and Segmentation 节点

#### 输入参数：
- **image**: 输入图像（ComfyUI张量格式，B×H×W×C）
- **model_name**: YOLO模型名称（从`ComfyUI/models/YOLO_MODEL/`文件夹自动加载）
- **confidence_threshold**: 检测置信度阈值 (0.0-1.0, 默认0.5)
- **filter_single_mask_by_class_id**: 按类别ID过滤合并掩码 (0-999, 默认0)
- **use_class_name_for_single_mask**: 启用类别名称过滤（布尔值）
  - 启用时：按类别名称过滤（不区分大小写）
  - 禁用时：使用类别ID过滤
- **class_name_for_single_mask**: 目标类别名称（当启用名称过滤时）
  - 支持模糊匹配，如"person", "car", "dog"等
  - 必须与模型的类别名称匹配
- **smooth_masks**: 启用掩码平滑（布尔值）
  - 使用scipy的高斯滤波进行边缘平滑
- **smooth_sigma**: 平滑强度 (0.0-10.0, 默认1.0)
  - 值越大，平滑效果越强，掩码边缘越柔和
- **force_output_mask**: 强制输出掩码（布尔值，默认True）
  - 启用时：当分割模型无掩码时，使用边界框自动生成掩码
  - 禁用时：仅使用模型原生分割掩码，无回退机制

#### 输出：
- **PLOTTED_IMAGE**: 带检测框和标签的图像
- **SINGLE_COMBINED_MASK**: 单个合并掩码（可按类别过滤）
- **ALL_INDIVIDUAL_MASKS**: 所有独立实例掩码
- **DETECTIONS_JSON**: 包含所有检测结果的JSON数据

### Masked Image Effects 节点

#### 输入参数：
- **image**: 输入图像（ComfyUI张量格式）
- **mask**: 输入掩码（H×W或1×H×W或1×1×H×W格式）
- **effect_type**: 效果类型
  - `gaussian_blur`: 高斯模糊效果
  - `mosaic`: 马赛克（像素化）效果
- **apply_mode**: 应用模式
  - `masked_area`: 效果应用于遮罩白色区域（前景）
  - `unmasked_area`: 效果应用于遮罩黑色区域（背景）
- **mosaic_pixel_size**: 马赛克像素块大小 (2-256, 默认16)
  - 值越大，像素块越大，马赛克效果越明显
- **gaussian_kernel_size**: 高斯模糊核大小 (3-255，必须为奇数，默认25)
  - 自动调整为奇数以满足要求
- **gaussian_sigma**: 高斯模糊标准差 (0.1-20.0, 默认5.0)
  - 控制模糊程度，值越大越模糊

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
1. **图像预处理**: 将ComfyUI张量(B×H×W×C)转换为PIL图像，移除批次维度
2. **模型推理**: 使用Ultralytics YOLOv8进行目标检测和分割推理
3. **结果处理**:
   - 解析YOLOv8检测结果，提取边界框、类别、置信度和分割掩码
   - BGR转RGB颜色空间转换，适配PIL和ComfyUI
   - 自动JSON序列化检测元数据
4. **高级掩码处理**:
   - 独立实例掩码提取和尺寸调整
   - 类别过滤的合并掩码生成
   - scipy.ndimage高斯滤波平滑处理
   - 边界框回退机制（当无分割掩码时）
5. **图像效果应用**:
   - 智能掩码尺寸适配（使用PIL LANCZOS重采样）
   - 区域选择性效果应用（前景/背景）
   - 多后端高斯模糊（torchvision优先，PIL降级）
   - 像素级马赛克效果实现
6. **输出生成**: 返回标准ComfyUI格式的图像和掩码张量

### 技术特性：
- **张量操作优化**: 使用torch.nn.functional.interpolate进行高效的掩码尺寸调整
- **内存管理**: 智能的设备分配（CPU/GPU），避免不必要的内存拷贝
- **错误处理**: 完善的异常捕获和日志记录机制
- **兼容性适配**: 自动处理不同输入格式和尺寸
- **性能优化**:
  - GPU加速支持（CUDA自动检测）
  - 向量化张量操作
  - 批量处理就绪
  - 内存映射优化

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
- 检查模型文件是否损坏或不完整（建议重新下载）
- 确认模型文件名拼写正确（区分大小写）
- 验证模型格式是否为.pt文件
- 查看ComfyUI终端日志，检查详细错误信息

**2. CUDA/GPU相关错误**
- 确保PyTorch CUDA版本与系统CUDA版本兼容
- 检查GPU内存是否充足（大模型需要更多显存）
- 尝试在CPU模式下运行：`force_output_mask`设为False以减少显存占用
- 更新显卡驱动到最新版本
- 检查CUDA Toolkit是否正确安装

**3. 掩码尺寸不匹配**
- 节点会自动处理尺寸不匹配问题（使用双线性插值）
- 检查ComfyUI日志中的尺寸调整信息
- 确保输入图像为有效的RGB/BGR格式
- 如需精确控制，可使用ComfyUI的Resize节点预处理

**4. 检测结果为空或无掩码**
- **降低置信度阈值**（从0.5降低到0.1-0.3）
- **检查模型类型**：
  - 使用-seg.pt模型获取分割掩码
  - 使用.pt模型仅获取边界框
- **验证目标类别**：
  - 确认图像中的目标与模型训练类别匹配
  - 检查类别名称大小写是否正确
- **启用force_output_mask**：当分割模型无掩码时使用边界框生成
- **检查图像质量**：光照、对比度、目标大小等因素

**5. 内存不足错误**
- 减小模型大小（n->s->m->l）
- 降低批次大小或图像分辨率
- 启用`force_output_mask`以减少内存占用
- 重启ComfyUI释放内存
- 监控系统资源使用情况

**6. 类别过滤不工作**
- **类别名称过滤**：
  - 确保类别名称与模型定义完全匹配
  - 尝试不同的大小写组合
  - 查看模型的类别列表：`model.names`
- **类别ID过滤**：
  - 确认类别ID在有效范围内（0-999）
  - 查看DETECTIONS_JSON输出确认实际检测到的类别

**7. 掩码平滑效果不明显**
- 增加`smooth_sigma`值（建议1.0-3.0）
- 确保输入掩码有清晰的边缘
- 检查掩码数据类型和范围（应为0-1浮点数）

**8. 效果节点尺寸适配问题**
- 节点自动处理尺寸不匹配
- 检查ComfyUI日志中的resize信息
- 确保mask输入为正确的张量格式
- 验证image和mask的设备一致性（CPU/GPU）

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
**兼容性**: ComfyUI
**Python**: >=3.11