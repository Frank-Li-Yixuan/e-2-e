# Plan B (Colab 高质量训练方案)

本文件描述如何在 Google Colab 或其它具备 NVIDIA GPU 的环境中运行 Plan B：使用 LoRA 微调 + 判别器 + 多损失，提高匿名质量与字符保真，同时保持隐私抑制。

## 1. 目标
- 更清晰且结构合理的匿名人脸与车牌。
- 降低 ArcFace 身份相似度 (ID Suppression)。
- 保持车牌字符分布合理 (OCR 可读性可控)。
- 改善整体视觉质量 (FID 降低)。

## 2. 环境需求
| 组件 | 推荐 | 备注 |
|------|------|------|
| GPU | T4 / A100 / RTX 3090+ | SD2 Inpaint fp16 + LoRA + 判别器训练舒适；SDXL 需 >=24GB |
| Python | 3.10 / 3.11 | 避免部分库在 3.13 尚无轮子 |
| PyTorch | >=2.0 (CUDA) | 支持 xFormers 与 `torch.compile` |
| xFormers | 最新 | 加速注意力与节省显存 |
| accelerate (可选) | 最新 | 分布式或混合精度管理 |

## 3. 安装示例 (Colab)
```bash
!pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
!pip install diffusers==0.28.0 transformers safetensors accelerate xformers
!pip install opencv-python-headless einops tqdm Pillow scikit-image pandas
!pip install onnxruntime-gpu easyocr
```
(若使用 SDXL inpaint 替换 `model_id` 为 `stabilityai/sdxl-inpaint` 并确保显存足够。)

## 4. 配置文件：`configs/planB_colab.yaml`
关键字段：
- `precision: fp16`：混合精度训练。
- `model.generator.diffusers.enable_xformers: true`：启用 xFormers。
- `model.finetune.use_lora: true` + `learnable_layers`: 仅微调 UNet 中部与末端上采样块，减少参数与过拟合风险。
- `model.discriminator.enabled: true`：使用判别器对匿名样本做对抗改进。
- 损失权重：`l1_weight`, `perceptual_weight`, `id_suppress_weight`, `adv_weight` 可根据观察调节。

### LoRA 可选扩展
将 `learnable_layers` 扩展加入跨注意力层：
```yaml
learnable_layers: ["unet.mid_block", "unet.up_blocks[-1]", "unet.down_blocks[0].attentions[0]"]
```
并控制 rank：在代码中包装参数时设置 `lora_rank` (需在 `generator_wrapper.py` 中添加 LoRA 注入逻辑)。

## 5. 运行训练
```bash
python -m src.train_joint --config configs/planB_colab.yaml --mode auto --max_steps 1000
```
阶段：
1. (可选) 预训练：使用 pseudo targets 或自重建暖身。
2. 联合交替：检测器 / 生成器 / 判别器 按 `alternating` 频率轮流更新。

## 6. 监控与输出
- 指标 CSV：`paths.outputs/metrics.csv`。包含 step, l1, perceptual, arcface_mean_sim, easyocr_plate_acc, fid, val_map。
- 中间样本：`val_samples/stepXXXX` 目录保存原图与匿名结果方便对比。
- Checkpoints：每 `checkpoint.save_every_steps` 保存一次各模块权重。

## 7. 调参与建议
| 目的 | 调节参数 | 说明 |
|------|----------|------|
| 提升字符清晰 | 增加 `plate_steps`, 减少 `plate_text_anchor_strength` | 锚定强度低保留合成文字更多细节 |
| 降低 ID 相似度 | 提高 `id_suppress_weight` | 过高可能导致人脸过度扭曲 |
| 改善整体逼真 | 调整 `adv_weight` + 适度提高判别器 capacity | 过强易出现模式塌缩或伪影 |
| 减少训练不稳定 | 梯度裁剪 / 降低学习率 | `lr.generator` 和 `lr.discriminator` 适度下调 |
| 减少显存占用 | 降低 batch / 关闭部分 LoRA 层 | 也可临时关闭判别器 |

## 8. 迁移 SDXL (可选)
- 将 `model.generator.diffusers.model_id` 改为 SDXL inpaint。
- 显存压力增大：需调低 batch 或 steps。
- Prompt 可更简洁，SDXL 语义更丰富；注意 token 长度仍需控制避免截断。

## 9. 常见问题
| 问题 | 原因 | 解决 |
|------|------|------|
| 训练速度慢 | GPU 性能不足 / I/O 阻塞 | 降低 steps, 使用更小分辨率，启用 xFormers |
| 判别器不稳定 | adv 权重过高 | 降低 `adv_weight` 或使用梯度裁剪 |
| 身份相似度下降有限 | ID 抑制权重过低 | 逐步增加 `id_suppress_weight` (0.1→0.25) |
| 车牌字符漂移 | 锚定字体比率不合适 | 调整 `plate_text_font_ratio_min/max` |

## 10. 后续增强想法
- 加入 Style Adapter：不同国家车牌风格权重组合微调。
- 使用 OCR Loss：对生成字符进行可读性约束（需合成标签，与匿名冲突时需平衡权重）。
- 分离人脸与车牌两套 LoRA：细粒度控制域适应。

## 11. 最小 Colab 启动单元格示例
```python
%cd /content
!git clone <your_repo_url> anony-project
%cd anony-project
!pip install -r requirements.txt
!pip install diffusers==0.28.0 transformers xformers accelerate safetensors onnxruntime-gpu easyocr
# 写入 paths 覆盖（推荐使用 unified COCO；脚本会优先使用 data/unified/unified_*.json，若缺失则尝试合并 widerface+pp4av）
%run scripts/colab_paths_autoset.py --mode auto
# 运行训练：当前版本的训练入口仅接收一个 config，如需合并 overlay，可将 overlay 中 paths 内容手动拷贝进 configs/planB_colab.yaml 的 paths 段落，或替换为你自己的 config 副本。
!python -m src.train_joint --config configs/planB_colab.yaml --mode auto --max_steps 500
```

## 12. 安全与隐私注意
- 对真实人脸做匿名训练时，ArcFace ID 相似度记录仅用于 aggregate，不保存可逆特征。
- 生成的匿名图像不应反推原身份；避免引入可逆嵌入。

---
若你需要：我可以继续为 LoRA 注入写一个简化的辅助模块或在 README 中添加 LoRA 参数说明。请告诉我下一步。