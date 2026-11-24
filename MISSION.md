# Mission

## Why I am learning this

Build an autonomous F1 racing car that sees the track through a camera and drives itself.
To do that I need to understand:
- How a camera sees the world (geometry, perspective, pixel math)
- How to find lane lines in an image (colour, shape, position)
- How to turn pixels into steering commands (control errors)
- How to train an agent to drive using those signals (PPO reinforcement learning)

## What I want to be able to do

1. Look at a camera frame and explain exactly how the detector finds the centre of the track
2. Write the vision pipeline from scratch — mask, scan, centre, errors
3. Understand what PPO is doing during training and why it works
4. Modify the pipeline and know what will break

## Learning path

| Phase | Topic | Labs |
|-------|-------|------|
| 1 | Classical computer vision | 01–08 |
| 2 | PyTorch fundamentals | new labs |
| 3 | Neural CV (replace classical blocks) | new labs |
| 4 | PPO reinforcement learning | training loop |
