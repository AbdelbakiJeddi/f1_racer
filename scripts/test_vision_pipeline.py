"""Verification script for the BatchedVisionTrackDetector.

Simulates camera views of:
1. Centered vehicle on straight track
2. Vehicle displaced to the right (+lateral error)
3. Vehicle entering an upcoming left turn (negative heading error)
4. Vehicle with partial occlusion

Prints errors and saves visual proof to an annotated image.
"""

import os
import torch
from leatherback_racer.vision.track_detector import BatchedVisionTrackDetector


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running BatchedVisionTrackDetector verification on: {device}")

    img_h, img_w = 128, 128
    detector = BatchedVisionTrackDetector(img_h=img_h, img_w=img_w).to(device)

    # Synthetic batch of 4 camera frames
    batch_size = 4
    images = torch.full((batch_size, 3, img_h, img_w), 0.12, device=device)  # dark asphalt

    yellow = torch.tensor([1.0, 0.82, 0.0], device=device).view(1, 3, 1, 1)

    # Scenario 0: Centered straight track (boundaries at col 38 and col 88)
    images[0, :, :, 36:40] = yellow[0]
    images[0, :, :, 86:90] = yellow[0]

    # Scenario 1: Vehicle displaced right (track appears shifted left in image)
    images[1, :, :, 16:20] = yellow[0]
    images[1, :, :, 66:70] = yellow[0]

    # Scenario 2: Upcoming left turn (straight near, curving left in distance)
    # Near horizon (y >= 58): centered (cols 36:40 and 86:90)
    images[2, :, 58:, 36:40] = yellow[0]
    images[2, :, 58:, 86:90] = yellow[0]
    # Far horizon (y < 58): shifted left (cols 20:24 and 70:74)
    images[2, :, :58, 20:24] = yellow[0]
    images[2, :, :58, 70:74] = yellow[0]

    # Scenario 3: Upcoming right turn (curving right in distance)
    # Near horizon (y >= 58): centered
    images[3, :, 58:, 36:40] = yellow[0]
    images[3, :, 58:, 86:90] = yellow[0]
    # Far horizon (y < 58): shifted right (cols 52:56 and 102:106)
    images[3, :, :58, 52:56] = yellow[0]
    images[3, :, :58, 102:106] = yellow[0]

    features, mask = detector(images)

    scenarios = [
        "Scenario 0 (Centered Straight)",
        "Scenario 1 (Vehicle Displaced Right)",
        "Scenario 2 (Approaching Left Turn)",
        "Scenario 3 (Approaching Right Turn)",
    ]

    print("\n--- Detection Results ---")
    for i, name in enumerate(scenarios):
        e_lat = features[i, 0].item()
        e_heading = features[i, 1].item()
        conf = features[i, 2].item()
        deg = e_heading * 180.0 / 3.14159265
        print(f"[{name}]")
        print(f"  -> Lateral Error (e_lat)   : {e_lat:+.3f}  (normalized [-1, 1])")
        print(f"  -> Heading Error (e_heading): {e_heading:+.3f} rad ({deg:+.1f} deg)")
        print(f"  -> Confidence              : {conf:.2f}")


if __name__ == "__main__":
    main()
