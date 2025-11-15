# 🚗 Capstone Design - Mini Autonomous Car

**영진전문대 글로벌시스템융합과 1학년 캡스톤디자인(I) 프로젝트**  
딥러닝과 임베디드 시스템을 활용한 **미니 자율주행 자동차 제작**

---

## 🎯 프로젝트 목표
- 카메라 입력 기반 **차선 인식 및 자율주행 (Lane Keeping)**
  → 이전 프로젝트 결과 영상: https://www.youtube.com/watch?v=0_gV1N-Q7Gw
- 팀별 **확장 기능 제안 및 구현**  
  ↳ *예: 신호등 인식, 장애물 감지, 속도 제어 등*

---

## 📂 주요 디렉토리

- [training/](training)  
  → 모델 학습, 데이터 전처리 및 Export  

- [inference/](inference/)  
  → Jetson Nano 추론 및 주행 제어 코드  

- [data-collector/](data-collector/)  
  → 카메라 + 자동차 제어 기반 데이터 수집 프로그램  
  - [hardware/](data-collector/hardware/) : 서보모터/DC모터 제어, 핀맵, 전원 관리  
  - [camera/](data-collector/camera/) : OpenCV 카메라 제어  

- [docs/](docs/)  
  → 프로젝트 문서, 다이어그램, 보고서  

---



