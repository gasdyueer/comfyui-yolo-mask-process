import folder_paths
from PIL import Image, ImageFilter
import numpy as np
from ultralytics import YOLO
import torch
import os
import json
import scipy.ndimage as ndi

def smooth_mask(mask, sigma):
    if sigma <= 0:
        return mask
    mask_np = mask.cpu().numpy()
    smoothed = ndi.gaussian_filter(mask_np.astype(np.float32), sigma=sigma)
    threshold = np.max(smoothed) / 2
    smoothed = np.where(smoothed >= threshold, 1.0, 0.0)
    return torch.from_numpy(smoothed).to(mask.device).to(mask.dtype)

def _prepare_image_for_inference(image_tensor):
    """将输入张量转换为PIL图像"""
    _, H, W, _ = image_tensor.shape
    image_np = image_tensor.cpu().numpy().squeeze(0)  # 移除批次维度 (BATCH, H, W, C) -> (H, W, C)
    pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
    return pil_image, H, W

def _load_yolov8_model(model_name):
    """加载YOLOv8模型"""
    yolov8_model_dir = folder_paths.get_folder_paths("YOLO_MODEL")[0]
    model_path = os.path.join(yolov8_model_dir, model_name)

    print(f'Loading YOLOv8 model from: {model_path}')
    model = YOLO(model_path)
    return model, model_path

def _initialize_outputs(image_tensor, H, W):
    """初始化所有输出"""
    output_image_tensor = image_tensor  # 默认输出原始图像，如果后续没有绘制
    empty_mask_hw = torch.zeros((H, W), dtype=torch.float32, device=image_tensor.device)
    empty_mask_nhw = torch.zeros((1, H, W), dtype=torch.float32, device=image_tensor.device)
    all_detected_objects_json = json.dumps({"detected_objects": [], "info": "No detections or segmentation masks found."})
    return output_image_tensor, empty_mask_hw, empty_mask_nhw, all_detected_objects_json

def _process_detection_results(results, model):
    """处理检测结果和元数据收集"""
    all_detected_objects_metadata = []
    output_image_tensor = None

    if results and len(results) > 0:
        r = results[0]  # 通常只有一个结果对象

        # 绘制结果图像
        im_array_with_predictions = r.plot()
        # 将 BGR numpy 数组转换为 RGB 给PIL处理，再转换为ComfyUI张量
        im_with_predictions = Image.fromarray(im_array_with_predictions[...,::-1])
        output_image_np = np.array(im_with_predictions).astype(np.float32) / 255.0
        output_image_tensor = torch.from_numpy(output_image_np).unsqueeze(0)  # 添加批次维度

        # 收集所有检测结果的元数据
        if r.boxes is not None and len(r.boxes) > 0:
            for i in range(len(r.boxes)):
                box = r.boxes[i]
                class_id = int(box.cls[0].item())
                conf_score = float(box.conf[0].item())

                obj_info = {
                    "class_id": class_id,
                    "class_name": model.names.get(class_id, "unknown"),
                    "confidence": conf_score,
                    "box_xyxy": box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                    "has_mask": False  # 默认没有Mask
                }

                # 如果存在masks，则更新状态
                if r.masks is not None and len(r.masks.data) > i:
                    obj_info["has_mask"] = True

                all_detected_objects_metadata.append(obj_info)

        all_detected_objects_json = json.dumps({
            "detected_objects": all_detected_objects_metadata,
            "info": f"Detected {len(all_detected_objects_metadata)} objects." if all_detected_objects_metadata else "No objects detected above confidence threshold."
        })
    else:
        print("Yolov8UnifiedNode: No detection results found.")
        all_detected_objects_json = json.dumps({"detected_objects": [], "info": "No detections or segmentation masks found."})

    return output_image_tensor, all_detected_objects_metadata, all_detected_objects_json

def _process_individual_masks(results, H, W, model_path):
    """处理所有独立Mask (N, H, W)"""
    empty_mask_nhw = torch.zeros((1, H, W), dtype=torch.float32, device=torch.device('cpu'))

    if results and results[0].masks is not None and len(results[0].masks.data) > 0:
        # masks.data 是 (N, H, W) 格式的张量
        all_individual_masks_tensor = results[0].masks.data.to(torch.float32)
        # 在某些情况下，YOLOv8的mask可能需要resize到原始图像大小
        # 如果模型输出的mask尺寸与输入图像pil_image的尺寸不完全一致
        # ComfyUI的Mask输入期望 HxW
        if (all_individual_masks_tensor.shape[1], all_individual_masks_tensor.shape[2]) != (H, W):
            print(f"Yolov8UnifiedNode: Resizing masks from {all_individual_masks_tensor.shape[1:]} to original image size {H, W}")
            # 使用torch.nn.functional.interpolate进行插值缩放
            # 假设 masks 是 N x 1 x H' x W'，需要先unsqueeze(1)
            all_individual_masks_tensor = torch.nn.functional.interpolate(
                all_individual_masks_tensor.unsqueeze(1),
                size=(H, W),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)  # 移除中间的channel维度
    else:
        all_individual_masks_tensor = empty_mask_nhw
        if model_path.endswith(("-seg.pt", "_seg.pt")):  # 检查是否是分割模型
              print("Yolov8UnifiedNode: Segmentation model used, but no masks detected or available in results.")
        else:
              print("Yolov8UnifiedNode: Model does not seem to support segmentation or no masks detected.")

    # 合并所有独立掩码
    if all_individual_masks_tensor.shape[0] > 0:
        all_individual_masks_tensor = torch.any(all_individual_masks_tensor, dim=0).unsqueeze(0)  # (H, W) -> (1, H, W)

    return all_individual_masks_tensor

def _process_single_combined_mask(results, H, W, filter_single_mask_by_class_id,
                                 use_class_name_for_single_mask, class_name_for_single_mask, model):
    """处理单个合并Mask (H, W)"""
    empty_mask_hw = torch.zeros((H, W), dtype=torch.float32, device=torch.device('cpu'))

    if results and results[0].masks is not None and len(results[0].masks.data) > 0:
        masks = results[0].masks.data  # (N, H, W)
        boxes = results[0].boxes       # Boxes object
        clss = boxes.cls               # 类别ID张量 (N,)

        target_lookup_id = None
        if use_class_name_for_single_mask:
            # 建立类别名称到ID的映射，并处理大小写
            name_to_id = {v.lower(): k for k, v in model.names.items()}
            input_class_name_lower = class_name_for_single_mask.lower().strip()

            if input_class_name_lower in name_to_id:
                target_lookup_id = name_to_id[input_class_name_lower]
                print(f"Yolov8UnifiedNode: Filtering single combined mask by class name '{class_name_for_single_mask}' (mapped to ID: {target_lookup_id}).")
            else:
                print(f"Yolov8UnifiedNode: Warning: Class name '{class_name_for_single_mask}' not found in model's recognized names. Single combined mask will be empty.")
                # 此时 target_lookup_id 仍是 None，会返回空mask
        else:
            target_lookup_id = filter_single_mask_by_class_id
            print(f"Yolov8UnifiedNode: Filtering single combined mask by class ID {filter_single_mask_by_class_id}.")

        # 如果找到了目标ID
        if target_lookup_id is not None:
            # 查找指定类别ID的索引
            target_indices = torch.where(clss == target_lookup_id)[0]

            if len(target_indices) > 0:
                # 提取并合并对应类别的mask
                target_masks = masks[target_indices]
                # 使用 torch.any 合并所有mask，并确保输出是 torch.float32 类型
                single_combined_mask_output = torch.any(target_masks, dim=0).to(torch.float32)
                # 如果mask尺寸与输入图像不一致，调整大小
                if single_combined_mask_output.shape != (H, W):
                    print(f"Yolov8UnifiedNode: Resizing single combined mask from {single_combined_mask_output.shape} to image size {H, W}")
                    single_combined_mask_output = torch.nn.functional.interpolate(
                        single_combined_mask_output.unsqueeze(0).unsqueeze(0),
                        size=(H, W),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze().to(torch.float32)
            else:
                single_combined_mask_output = empty_mask_hw
                print(f"Yolov8UnifiedNode: No objects of target filter criteria detected for single combined mask.")
        else:
            single_combined_mask_output = empty_mask_hw
            print("Yolov8UnifiedNode: Invalid filter criteria for single combined mask. Returning empty mask.")
    else:
        single_combined_mask_output = empty_mask_hw
        print("Yolov8UnifiedNode: No masks available for single combined mask generation.")

    return single_combined_mask_output

def _apply_smoothing(masks_tensor, single_mask, smooth_masks, smooth_sigma):
    """应用平滑到掩码如果启用"""
    if smooth_masks and smooth_sigma > 0:
        if masks_tensor.numel() > 0:
            smoothed_individual = []
            for mask in masks_tensor:
                smoothed_individual.append(smooth_mask(mask, smooth_sigma))
            masks_tensor = torch.stack(smoothed_individual)
        if single_mask.numel() > 0:
            single_mask = smooth_mask(single_mask, smooth_sigma)

    return masks_tensor, single_mask

# 设置YOLO模型的路径
# 确保这个路径设置是正确的，ComfyUI通常会自动管理这一部分
# 或者用户需要手动将YOLO模型（例如 yolov8n.pt, yolov8n-seg.pt）放入 models/YOLO_MODEL 文件夹
folder_paths.folder_names_and_paths["YOLO_MODEL"] = ([os.path.join(folder_paths.models_dir, "YOLO_MODEL")], folder_paths.supported_pt_extensions)

class YoloDetectionAndSegmentation:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (folder_paths.get_filename_list("YOLO_MODEL"), ),
                "confidence_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                # 用于过滤单个合并Mask的选项
                "filter_single_mask_by_class_id": ("INT", {"default": 0, "min": 0}),
                "use_class_name_for_single_mask": ("BOOLEAN", {"default": False, "label_on": "Enable Class Name Filter", "label_off": "Use Class ID Filter"}),
                "class_name_for_single_mask": ("STRING", {"default": "", "multiline": False}),
                # 平滑掩码选项
                "smooth_masks": ("BOOLEAN", {"default": False, "label_on": "Enable Mask Smoothing", "label_off": "Disable Mask Smoothing"}),
                "smooth_sigma": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            },
        }

    # 返回带标注的图像，一个针对特定类别的合并Mask，一个所有对象的Mask批次，以及一个包含所有检测信息的JSON
    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "JSON")
    RETURN_NAMES = ("PLOTTED_IMAGE", "SINGLE_COMBINED_MASK", "ALL_INDIVIDUAL_MASKS", "DETECTIONS_JSON")
    FUNCTION = "process_yolov8"
    CATEGORY = "yolo-mask-process" # 全新分类，表明是综合节点

    def process_yolov8(self, image, model_name, confidence_threshold,
                         filter_single_mask_by_class_id, use_class_name_for_single_mask,
                         class_name_for_single_mask, smooth_masks, smooth_sigma):

        # 步骤1: 图像预处理
        pil_image, H, W = _prepare_image_for_inference(image)

        # 步骤2: 模型加载和推理
        model, model_path = _load_yolov8_model(model_name)
        results = model(pil_image, conf=confidence_threshold)

        # 步骤3: 初始化输出
        output_image_tensor, empty_mask_hw, empty_mask_nhw, all_detected_objects_json = _initialize_outputs(image, H, W)

        # 步骤4: 处理检测结果
        output_image_tensor, all_detected_objects_metadata, all_detected_objects_json = _process_detection_results(results, model)

        # 步骤5: 处理独立mask
        all_individual_masks_tensor = _process_individual_masks(results, H, W, model_path)

        # 步骤6: 处理单个合并mask
        single_combined_mask_output = _process_single_combined_mask(
            results, H, W, filter_single_mask_by_class_id,
            use_class_name_for_single_mask, class_name_for_single_mask, model
        )

        # 步骤7: 应用平滑
        all_individual_masks_tensor, single_combined_mask_output = _apply_smoothing(
            all_individual_masks_tensor, single_combined_mask_output, smooth_masks, smooth_sigma
        )

        return (output_image_tensor, single_combined_mask_output, all_individual_masks_tensor, all_detected_objects_json)


# Mask Process Node - 第二个文件的内容
from PIL.Image import Resampling
try:
    import torchvision.transforms as T
except ImportError:
    T = None


class MaskedImageEffectsNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "effect_type": (
                    [
                        "gaussian_blur",
                        "mosaic",
                    ],
                    {"default": "gaussian_blur"}
                ),
                "apply_mode": ( # 控制效果应用于Mask区域还是非Mask区域
                    [
                        "masked_area",    # 效果应用于mask白色区域
                        "unmasked_area",  # 效果应用于mask黑色区域
                    ],
                    {"default": "masked_area"}
                ),
                # 马赛克效果的参数
                "mosaic_pixel_size": ("INT", {"default": 16, "min": 2, "max": 256, "step": 1, }),
                # 高斯模糊效果的参数
                "gaussian_kernel_size": ("INT", {"default": 25, "min": 3, "max": 255, "step": 2, }), # 高斯核大小，必须是奇数
                "gaussian_sigma": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 20.0, "step": 0.1, }), # 高斯函数的标准差
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_image_effect"
    CATEGORY = "yolo-mask-process" # 放在新的分类下

    def apply_image_effect(self,
                            image: torch.Tensor,
                            mask: torch.Tensor,
                            effect_type: str,
                            apply_mode: str,
                            mosaic_pixel_size: int,
                            gaussian_kernel_size: int,
                            gaussian_sigma: float):

        # 基础数据准备
        img_np = image.squeeze(0).cpu().numpy() # (H, W, C) float32 [0,1]
        # 确保mask和image的尺寸匹配
        # mask通常是(H, W)或(1, H, W)，但也可能有(1, 1, H, W)
        if mask.dim() == 4 and mask.shape[0] == 1 and mask.shape[1] == 1:
            mask_np = mask.squeeze(0).squeeze(0).cpu().numpy() # (H, W) float32 [0,1]
        elif mask.dim() == 3 and mask.shape[0] == 1:
            mask_np = mask.squeeze(0).cpu().numpy() # (H, W) float32 [0,1]
        elif mask.dim() == 2:
            mask_np = mask.cpu().numpy() # (H, W) float32 [0,1]
        else:
            raise ValueError(f"Unsupported mask shape: {mask.shape}. Expected (H, W) or (1, H, W) or (1, 1, H, W).")


        H_img, W_img, C = img_np.shape
        H_mask, W_mask = mask_np.shape

        # 如果mask和image尺寸不匹配，对mask进行resize
        if H_img != H_mask or W_img != W_mask:
            print(f"MaskedImageEffectsNode: Resizing mask from {H_mask}x{W_mask} to image size {H_img}x{W_img}.")
            mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
            mask_pil = mask_pil.resize((W_img, H_img), Resampling.LANCZOS)
            mask_np = np.array(mask_pil).astype(np.float32) / 255.0


        output_img_np = np.copy(img_np) # 用于存储最终效果的副本

        # 准备二进制Mask
        binary_mask = (mask_np > 0.5) # mask白色区域为True

        # 如果 apply_mode 是 "unmasked_area"，则反转 mask
        if apply_mode == "unmasked_area":
            binary_mask = ~binary_mask
            print(f"MaskedImageEffectsNode: Applying {effect_type} to UNMASKED area.")
        else: # masked_area
            print(f"MaskedImageEffectsNode: Applying {effect_type} to MASKED area.")


        # --- 应用马赛克效果 ---
        if effect_type == "mosaic":
            print(f"MaskedImageEffectsNode: Applying Mosaic with pixel size {mosaic_pixel_size}.")
            for y in range(0, H_img, mosaic_pixel_size):
                for x in range(0, W_img, mosaic_pixel_size):
                    y_end = min(y + mosaic_pixel_size, H_img)
                    x_end = min(x + mosaic_pixel_size, W_img)

                    # 检查当前块内是否有任何像素落在目标Mask区域内
                    # 如果有，则对整个块计算平均颜色
                    if np.any(binary_mask[y:y_end, x:x_end]):
                        block = img_np[y:y_end, x:x_end]
                        if block.size > 0:
                            avg_color = np.mean(block, axis=(0, 1))
                            # 仅将这个平均颜色值填充到该块中属于 binary_mask 的像素
                            # 使用 advanced indexing 来只修改被mask覆盖的像素
                            output_img_np[y:y_end, x:x_end][binary_mask[y:y_end, x:x_end]] = avg_color

            final_output_img_tensor = torch.from_numpy(output_img_np).unsqueeze(0)

        # --- 应用高斯模糊效果 ---
        elif effect_type == "gaussian_blur":
            if T is not None:
                # 使用torchvision进行高斯模糊
                # 确保 kernel_size 是奇数
                if gaussian_kernel_size % 2 == 0:
                    gaussian_kernel_size += 1
                    print(f"MaskedImageEffectsNode: Adjusted gaussian_kernel_size to {gaussian_kernel_size} (must be odd).")

                print(f"MaskedImageEffectsNode: Applying Gaussian Blur with kernel size {gaussian_kernel_size}, sigma {gaussian_sigma}.")

                img_tensor_bchw = image.permute(0, 3, 1, 2) # (B, H, W, C) -> (B, C, H, W)
                # mask (H, W) -> (B, 1, H, W) 用于广播
                # target_mask 用于在 (B, C, H, W) 图像上进行逐像素混合
                # 这里的 mask_np 已经是根据 image 尺寸调整过的
                target_mask_b1hw = (torch.from_numpy(binary_mask).to(img_tensor_bchw.device).to(torch.float32)).unsqueeze(0).unsqueeze(0)

                # 创建高斯模糊变换
                gaussian_blur_transform = T.GaussianBlur(
                    kernel_size=(gaussian_kernel_size, gaussian_kernel_size),
                    sigma=(gaussian_sigma, gaussian_sigma)
                )

                # 对整个图像应用模糊
                blurred_img_tensor_bchw = gaussian_blur_transform(img_tensor_bchw)

                # 混合原始图像和模糊图像
                # pixel_wise_mask: 在目标区域为1，非目标区域为0
                # output = blurred * pixel_wise_mask + original * (1 - pixel_wise_mask)
                final_output_img_tensor_bchw = (blurred_img_tensor_bchw * target_mask_b1hw) + \
                                               (img_tensor_bchw * (1 - target_mask_b1hw))

                final_output_img_tensor = final_output_img_tensor_bchw.permute(0, 2, 3, 1) # (B, C, H, W) -> (B, H, W, C)
            else:
                # Fallback to PIL for Gaussian Blur
                print(f"MaskedImageEffectsNode: Applying Gaussian Blur with PIL (torchvision not available), sigma {gaussian_sigma}.")

                # Convert to PIL
                img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
                blurred_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=gaussian_sigma))
                blurred_np = np.array(blurred_pil).astype(np.float32) / 255.0

                # Blend with mask
                output_img_np = blurred_np * binary_mask[..., np.newaxis] + img_np * (1 - binary_mask[..., np.newaxis])
                final_output_img_tensor = torch.from_numpy(output_img_np).unsqueeze(0)

        else:
            print(f"MaskedImageEffectsNode: Unknown effect_type '{effect_type}'. Returning original image.")
            final_output_img_tensor = image # 未知效果，返回原图

        return (final_output_img_tensor,)


# --------------------------------------------------------------------
# 节点注册
# --------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "YoloDetectionAndSegmentation": YoloDetectionAndSegmentation, # 综合节点
    "MaskedImageEffects": MaskedImageEffectsNode, # 新的综合效果节点
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YoloDetectionAndSegmentation": "YOLO Detection", # 综合节点显示名称改为YOLO Detection
    "MaskedImageEffects": "Mask Process", # 新的综合效果节点显示名称改为Mask Process
}