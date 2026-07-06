# Ellipse Pillar Aligner

타원 단면 기둥을 위에서 관측하여 **yaw angle만 계산**하고, 로봇팔이 해당 yaw만큼 회전 정렬할 수 있도록 명령값을 생성하는 독립 프로젝트입니다.

기존 `rs_local` 미션형 데모와 분리된 별도 프로젝트입니다.

## 목적

1. RealSense RGB 영상에서 타원을 검출한다.
2. 타원의 중심, 장축, 단축, yaw angle을 계산한다.
3. 목표 yaw 방향과 비교하여 회전해야 할 각도를 계산한다.
4. 타원보다 큰 ROI를 잡고, 배경 Depth와 현재 Depth 차이를 계산한다.
5. 웹 UI에서 실시간 결과를 확인한다.
6. 로봇팔 제어부는 안전한 preview stub으로 둔다.

## 폴더 구조

```text
ellipse_aligner/
├── app.py                  # Flask 서버, MJPEG 스트림, API
├── config.py               # 카메라/각도/ROI 설정
├── realsense_camera.py     # RealSense 초기화, 프레임 획득, 배경 캡처
├── vision_ellipse.py       # RGB 기반 타원 검출, 장축/단축/yaw 계산
├── depth_roi.py            # 배경 대비 Depth ROI 높이 계산
├── overlay.py              # 실시간 화면 overlay 표시
├── math_utils.py           # 각도 정규화, 목표각 계산
├── robot_interface.py      # 로봇팔 명령 preview stub
├── test_realsense.py       # RealSense 연결 사전 테스트
├── requirements.txt        # 의존성
├── run.sh                  # 실행 스크립트
└── templates/
    └── index.html          # 웹 UI
```

## 실행 방법

```bash
cd ellipse_aligner
pip install -r requirements.txt
python3 test_realsense.py
python3 app.py
```

브라우저에서 접속:

```text
http://localhost:5001
```

## 기본 사용 순서

1. `START`를 누른다.
2. 물체가 없는 상태에서 `배경 Depth 캡처`를 누른다.
3. 타원기둥을 카메라 아래에 둔다.
4. 화면에서 타원, 장축, yaw, rotate command를 확인한다.
5. `로봇 명령 미리보기`로 현재 명령값을 확인한다.

## 각도 convention

- `yaw_deg`: RGB 영상에서 검출된 타원 장축의 방향입니다.
- `target_yaw_deg`: 정렬하고 싶은 목표 방향입니다. 기본값은 90도, 즉 화면 세로 방향입니다.
- `rotate_deg`: 현재 yaw에서 target yaw로 가기 위해 필요한 최소 회전각입니다.

타원은 180도 대칭이므로 각도는 `-90° ~ +90°` 범위로 정규화합니다.

## 로봇팔 연결

현재 `robot_interface.py`는 안전상 실제 로봇을 움직이지 않는 preview stub입니다.
실제 로봇팔을 연결할 때는 다음을 추가해야 합니다.

1. 카메라 픽셀 좌표 `(cx, cy)` → 로봇 XY 좌표 변환
2. 그리퍼 접근 높이 설정
3. Z축 회전 명령 적용
4. pick/place sequence 작성
5. 비상 정지 조건 추가

```python
robot_rotate_deg = ROBOT_ROTATION_SIGN * rotate_deg + ROBOT_ROTATION_OFFSET_DEG
```

`ROBOT_ROTATION_SIGN`과 `ROBOT_ROTATION_OFFSET_DEG`는 `config.py`에서 보정합니다.
