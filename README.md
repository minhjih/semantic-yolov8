# semantic-yolov8
semantic compression for yolov8

## run.py 실행 프로세스
- 위치: `/home/aislab/Research/Semantic/run.py` (이 디렉터리에서 실행 권장)
- 기본 실행: `python run.py`  
  - epochs=100  
  - Train_Dist grid: student_layer_num {3,4,5}, compression_ratio {2,4,8}  
  - Train_Sem grid: compression_ratio {2,4,8}, lambda 세트 {(1,1,1), (0.5,1,1), (1,0.5,1), (1,1,0.5), (1,0.5,0.5), (0.5,1,0.5), (0.5,0.5,1)}
- 옵션:
  - `--task dist|sem|all` 실험 범위 선택
  - `--epochs N` 반복 수 오버라이드
  - `--dist-layers ...`, `--dist-compressions ...`, `--sem-compressions ...`, `--sem-lambdas ...` 로 커스텀 그리드 지정
- 동작:
  - 각 조합 실행 시 `run_logs/<tag>.log` 로 로그 저장
  - 각 실험 종료 직후 `results.txt` 에 `avg_loss`, `Box Diff`, `Class Diff` 등을 한 줄씩 append (실시간 확인 가능)
  - 모든 실행 후 `results.txt` 에 정렬/베스트 요약을 다시 작성
  - 학습 스크립트에서 저장하는 모델 파일명은 실험 `tag` 가 suffix 로 붙어 충돌 방지 (예: `dist_temp_<tag>.pth`, `sem_temp_<tag>.pth`)
