"""Quick RealSense test for Ellipse Pillar Aligner."""
import sys
import time


def main():
    print("=" * 60)
    print("Ellipse Aligner RealSense Test")
    print("=" * 60)

    try:
        import pyrealsense2 as rs
        import numpy as np
        print(f"[1] pyrealsense2 import: PASS ({rs.__version__})")
    except Exception as e:
        print(f"[1] pyrealsense2 import: FAIL - {e}")
        return 1

    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        print("[2] camera connection: FAIL - no RealSense device")
        print("    Check USB 3.0 port and run: lsusb | grep Intel")
        return 1
    dev = devices[0]
    print(f"[2] camera connection: PASS - {dev.get_info(rs.camera_info.name)}")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    try:
        profile = pipeline.start(config)
        print("[3] pipeline start: PASS")
    except Exception as e:
        print(f"[3] pipeline start: FAIL - {e}")
        return 1

    try:
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        print(f"[4] depth scale: {depth_scale:.6f}")
        print(f"[4] intrinsics: fx={intr.fx:.1f}, fy={intr.fy:.1f}, cx={intr.ppx:.1f}, cy={intr.ppy:.1f}")

        align = rs.align(rs.stream.color)
        ok = 0
        for i in range(5):
            frames = pipeline.wait_for_frames(timeout_ms=2000)
            aligned = align.process(frames)
            color = aligned.get_color_frame()
            depth = aligned.get_depth_frame()
            if color and depth:
                c = np.asanyarray(color.get_data())
                d = np.asanyarray(depth.get_data())
                center = depth.get_distance(320, 240)
                print(f"[5] frame {i+1}: color={c.shape}, depth={d.shape}, center={center:.3f} m")
                ok += 1
            time.sleep(0.05)
        print(f"[5] frame acquisition: {'PASS' if ok >= 3 else 'FAIL'} ({ok}/5)")
    finally:
        pipeline.stop()
        print("[6] pipeline stop: PASS")

    return 0 if ok >= 3 else 1


if __name__ == "__main__":
    sys.exit(main())
