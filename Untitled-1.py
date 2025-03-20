import os
from ultralytics import YOLO

def pose_detection(m, image_path, output_path):
    # YOLO 모델 초기화
    model = YOLO(m)
    
    # 이미지 처리
    results = model(image_path)  # 결과가 리스트로 반환될 수 있음
    
    # 출력 경로 생성 확인
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 각 결과를 저장
    for i, result in enumerate(results):
        save_path = os.path.join(output_path, f"result_{i}.jpg")
        result.plot()  # 결과 이미지에 예측된 데이터 표시
        result.save(save_path)
        print(f"Result {i + 1} saved to {save_path}")

if __name__ == '__main__':
    pose_detection(
        m='yolo11n-pose.pt',
        image_path=r'D:\INFO SECURITY\Pose_detection\judo.jpg',
        output_path=r'D:\INFO SECURITY\Pose_detection\results'
    )
