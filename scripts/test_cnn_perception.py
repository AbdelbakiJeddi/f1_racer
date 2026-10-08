import os
import sys
import time
import torch
import numpy as np
from PIL import Image, ImageDraw

# Add src to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from leatherback_racer.vision.transforms import VisionPreprocessor, VisualDomainRandomizer
from leatherback_racer.vision.frame_stack import BatchedFrameStack
from leatherback_racer.models.cnn_encoders import NatureCNNEncoder, SpatialSoftmaxEncoder
from leatherback_racer.models.policy import AsymmetricActorCriticPolicy


def main():
    print("\n" + "=" * 70)
    print(" LEATHERBACK RACER: CNN PERCEPTION & ARCHITECTURE VERIFICATION")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Target Device: {device}")

    # 1. Load sample captured onboard camera frame
    sample_img_paths = [
        "isaacsim/scenes/arena_circuit_01/wavy_onboard_view.png",
        "isaacsim/scenes/arena_wavy_circle/wavy_onboard_view.png",
    ]
    img_path = None
    for p in sample_img_paths:
        if os.path.exists(p):
            img_path = p
            break

    if img_path is None:
        print("[WARNING] No real rendered camera image found. Creating synthetic RGB track image.")
        # Synthetic fallback
        img_np = np.zeros((480, 640, 3), dtype=np.uint8)
        img_np[200:, :, :] = 30  # dark road
        img_np[200:, 200:220, :] = [255, 210, 0]  # left yellow line
        img_np[200:, 420:440, :] = [255, 210, 0]  # right yellow line
        raw_pil = Image.fromarray(img_np)
    else:
        print(f"  Loaded Camera Image: {img_path}")
        raw_pil = Image.open(img_path).convert("RGB")

    raw_tensor = torch.from_numpy(np.array(raw_pil)).permute(2, 0, 1).unsqueeze(0).to(device) # (1, 3, 480, 640)
    print(f"  Input Tensor Shape: {raw_tensor.shape}, dtype: {raw_tensor.dtype}")

    # 2. Test VisionPreprocessor
    preprocessor_rgb = VisionPreprocessor(output_size=(84, 84), color_mode="rgb").to(device)
    preprocessor_yellow = VisionPreprocessor(output_size=(84, 84), color_mode="yellow_contrast").to(device)

    proc_rgb = preprocessor_rgb(raw_tensor)
    proc_yellow = preprocessor_yellow(raw_tensor)

    print("\n[STEP 1] Preprocessor Output:")
    print(f"  RGB Preprocessed:    shape={proc_rgb.shape}, min={proc_rgb.min():.2f}, max={proc_rgb.max():.2f}")
    print(f"  Yellow Channel:      shape={proc_yellow.shape}, min={proc_yellow.min():.2f}, max={proc_yellow.max():.2f}")

    # 3. Test Visual Domain Randomizer
    randomizer = VisualDomainRandomizer().to(device)
    randomizer.train()
    rand_rgb = randomizer(proc_rgb)
    print(f"  Randomized RGB:      shape={rand_rgb.shape}")

    # 4. Test Batched Frame Stacking
    num_envs = 32
    frame_stack = BatchedFrameStack(num_envs=num_envs, stack_size=3, num_channels=3, height=84, width=84, device=device)
    dummy_batched = proc_rgb.repeat(num_envs, 1, 1, 1)
    frame_stack.reset(torch.arange(num_envs, device=device), dummy_batched)
    stacked_obs = frame_stack.step(dummy_batched)
    print(f"\n[STEP 2] Batched Frame Stack (k=3, N={num_envs}):")
    print(f"  Stacked Tensor Shape: {stacked_obs.shape} (Expected: (32, 9, 84, 84))")

    # 5. Test Nature-CNN Encoder
    nature_cnn = NatureCNNEncoder(in_channels=9, features_dim=128, input_size=(84, 84)).to(device)
    z_nature = nature_cnn(stacked_obs)
    nature_params = sum(p.numel() for p in nature_cnn.parameters() if p.requires_grad)
    print("\n[STEP 3] Nature-CNN Feature Extractor:")
    print(f"  Parameters:          {nature_params:,}")
    print(f"  Latent Embedding:    shape={z_nature.shape}")

    # 6. Test Spatial Softmax Keypoint Encoder
    spatial_encoder = SpatialSoftmaxEncoder(in_channels=9, num_keypoints=16, features_dim=128, input_size=(84, 84)).to(device)
    z_spatial, keypoints = spatial_encoder(stacked_obs)
    spatial_params = sum(p.numel() for p in spatial_encoder.parameters() if p.requires_grad)
    print("\n[STEP 4] Spatial-Softmax Keypoint Encoder:")
    print(f"  Parameters:          {spatial_params:,}")
    print(f"  Keypoints Tensor:    shape={keypoints.shape} (16 (x,y) landmarks)")
    print(f"  Latent Embedding:    shape={z_spatial.shape}")

    # 7. Test Asymmetric Actor-Critic Policy
    policy = AsymmetricActorCriticPolicy(
        in_channels=9,
        proprio_dim=2,
        action_dim=2,
        privileged_dim=10,
        encoder_type="nature_cnn",
        features_dim=128,
        input_size=(84, 84),
    ).to(device)
    total_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print("\n[STEP 5] Asymmetric Actor-Critic Policy:")
    print(f"  Total Parameters:    {total_params:,}")

    proprio = torch.zeros(num_envs, 2, device=device) # [steer, speed]
    actions, log_prob, _ = policy.get_action(stacked_obs, proprio)
    print(f"  Sampled Actions:     shape={actions.shape}, steer range=[{actions[:, 0].min():.2f}, {actions[:, 0].max():.2f}]")
    print(f"  Log Probabilities:   shape={log_prob.shape}")

    # Test PPO Privileged Critic Evaluation
    priv_state = torch.zeros(num_envs, 10, device=device)
    values, eval_log_prob, entropy = policy.evaluate_actions(stacked_obs, proprio, priv_state, actions)
    print(f"  Critic Value V(s):   shape={values.shape}")
    print(f"  Policy Entropy:      mean={entropy.mean().item():.3f}")

    # 8. Inference Throughput Benchmark (FPS)
    print("\n[STEP 6] GPU Inference Throughput Benchmark (Actor Only):")
    policy.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(50):
            _ = policy.actor(stacked_obs, proprio)
        if device.type == "cuda":
            torch.cuda.synchronize()

        n_iters = 200
        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = policy.actor(stacked_obs, proprio)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

    total_frames = n_iters * num_envs
    fps = total_frames / elapsed
    latency_ms = (elapsed / n_iters) * 1000.0
    print(f"  Batch Size:          {num_envs} environments")
    print(f"  Step Latency:        {latency_ms:.2f} ms per batch")
    print(f"  Throughput:          {fps:,.0f} frames / second on {device}")

    # 9. Save Annotated Visual Verification Composite Image
    os.makedirs("outputs", exist_ok=True)
    out_vis_path = "outputs/cnn_perception_verification.png"

    # Assemble visualization canvas
    rgb_84 = (proc_rgb[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    yellow_84 = (proc_yellow[0, 0].cpu().numpy() * 255).astype(np.uint8)
    yellow_rgb = np.stack([yellow_84, yellow_84, yellow_84], axis=-1)

    pil_rgb_84 = Image.fromarray(rgb_84).resize((256, 256), Image.NEAREST)
    pil_yellow = Image.fromarray(yellow_rgb).resize((256, 256), Image.NEAREST)
    pil_raw = raw_pil.resize((341, 256), Image.BILINEAR)

    # Plot spatial keypoints on top of yellow channel
    kp_sample = keypoints[0].detach().cpu().numpy().reshape(-1, 2)
    pil_kp = pil_yellow.copy()
    draw = ImageDraw.Draw(pil_kp)
    w_k, h_k = pil_kp.size
    for i, (kx, ky) in enumerate(kp_sample):
        px = int((kx + 1.0) * 0.5 * w_k)
        py = int((ky + 1.0) * 0.5 * h_k)
        draw.ellipse([px - 4, py - 4, px + 4, py + 4], fill=(255, 50, 50), outline=(255, 255, 255))

    # Canvas of 4 panes
    canvas = Image.new("RGB", (341 + 256 * 3, 290), color=(20, 20, 25))
    canvas.paste(pil_raw, (0, 30))
    canvas.paste(pil_rgb_84, (341, 30))
    canvas.paste(pil_yellow, (341 + 256, 30))
    canvas.paste(pil_kp, (341 + 512, 30))

    canvas_draw = ImageDraw.Draw(canvas)
    canvas_draw.text((10, 8), "1. Onboard Camera (640x480)", fill=(220, 220, 220))
    canvas_draw.text((351, 8), "2. ROI Hood Crop (84x84)", fill=(220, 220, 220))
    canvas_draw.text((351 + 256, 8), "3. Yellow-Line Filter Channel", fill=(220, 220, 220))
    canvas_draw.text((351 + 512, 8), "4. Spatial Keypoints (16 Landm.)", fill=(220, 220, 220))

    canvas.save(out_vis_path)
    print(f"\n[SUCCESS] Visual verification saved to: {out_vis_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
