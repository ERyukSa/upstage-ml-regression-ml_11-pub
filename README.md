# 교통 특성을 활용한 주택 가격 예측 모델

## ML_11조 - Upstage ML 회귀 경진대회 7위

<table>
  <tr>
    <td align="center"><img src="https://github.com/AIBootcamp13/upstage-ml-regression-ml_11/blob/wb2x/docs/team/images/AI13_홍상호.jpeg?raw=true" width="180" height="180"/></td>
    <td align="center"><img src="https://github.com/AIBootcamp13/upstage-ml-regression-ml_11/blob/wb2x/docs/team/images/AI13_신광명.jpg?raw=true" width="180" height="180"/></td>
    <td align="center"><img src="https://github.com/AIBootcamp13/upstage-ml-regression-ml_11/blob/wb2x/docs/team/images/AI13_최용비.png?raw=true" width="180" height="180"/></td>
    <td align="center"><img src="https://github.com/AIBootcamp13/upstage-ml-regression-ml_11/blob/wb2x/docs/team/images/AI13_이영준.png?raw=true" width="180" height="180"/></td>
    <td align="center"><img src="https://github.com/AIBootcamp13/upstage-ml-regression-ml_11/blob/wb2x/docs/team/images/AI13_김문수.jpg?raw=true" width="180" height="180"/></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/YOUR_GITHUB">AI13_홍상호</a></td>
    <td align="center"><a href="https://github.com/YOUR_GITHUB">AI13_신광명</a></td>
    <td align="center"><a href="https://github.com/YOUR_GITHUB">AI13_최용비</a></td>
    <td align="center"><a href="https://github.com/YOUR_GITHUB">AI13_이영준</a></td>
    <td align="center"><a href="https://github.com/YOUR_GITHUB">AI13_김문수</a></td>
  </tr>
  <tr>
    <td align="center">시각화, 데이터 검증</td>
    <td align="center">EDA, 전처리</td>
    <td align="center">교통 피처 병렬화 (Wb2x2)</td>
    <td align="center">모델링, 모델 최적화</td>
    <td align="center">특성 공학, 교통 관련 피처 생성</td>
  </tr>
</table>

## 0. 개요

저희 솔루션은 Upstage ML 회귀 경진대회에서 7위(RMSE: 12579.0120)를 달성했습니다. 고급 교통 특성 공학을 활용한 종합적인 부동산 가격 예측 접근 방식을 구현했습니다. 대중교통과 교통 허브와의 근접성을 측정하는 정교한 공간 특성을 만드는 데 중점을 두어 예측 정확도를 크게 향상시켰습니다.

### 개발 환경

- Python 3.9.21
- Ubuntu 20.04.6 LTS
- CUDA 11.8
- 라이브러리: pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn

### 필요 라이브러리

```
pandas>=1.3.5
numpy>=1.20.3
scikit-learn>=1.0.2
xgboost>=1.5.0
matplotlib>=3.5.1
seaborn>=0.11.2
```

### 시스템 사양

- **CPU**: AMD Ryzen Threadripper 3970X (32-Core)
- **메모리**: 251.62 GiB
- **저장소**: 5.46 TiB (Samsung SSD 970 EVO Plus 2TB + HGST HUS726T4TAL)
- **GPU**: 4x NVIDIA GeForce RTX 3090
- **네트워크**: 10 Gbps 연결

## 1. 대회 정보

### 개요

이 대회는 부동산 특성과 교통 접근성 정보를 포함한 데이터셋을 기반으로 한국의 주택 가격을 예측하는 데 초점을 맞추었습니다. 참가자들은 대중교통 근접성의 영향을 고려하면서 주택 가치를 정확하게 예측할 수 있는 회귀 모델을 구축하도록 도전받았습니다.

### 타임라인

- 2025년 5월 1일 - 대회 시작일
- 2025년 5월 15일 - 최종 제출 마감일
- 2025년 5월 16일 - 최종 결과 발표

## 2. 프로젝트 구성

### 디렉토리 구조

```
├── code
│   ├── XGBoost_refactor_RMSE_1894.8507                           # 리팩토링된 XGBoost 모델
│   ├── address_to_geo                                            # 주소 지오코딩 변환
│   └── xgboost_model_with_advanced_transportation_features       # 고급 교통 특성 모델
│       ├── XGBOOST_revised_transportation.ipynb                  # 주요 모델 학습 노트북
│       ├── outputs                                               # 모델 분석 및 시각화 결과
│       ├── cache                                                 # 캐시된 공간 인덱스 및 특성
│       │   └── transportation_features
│       │       ├── spatial_indices
│       │       ├── data_with_transport_features.pkl
│       │       └── hub_clustering.pkl
│       └── transportation_features_module                        # 교통 특성 모듈
│           ├── transport_features_module.py                      # 핵심 교통 특성 모듈
│           ├── transport_features_parallel.py                    # 병렬 처리 최적화
│           ├── transportation_clustering.py                      # 교통 허브 클러스터링
│           └── transport_feature_analysis.py                     # 교통 특성 분석
├── data
│   ├── train.csv                                                 # 원본 학습 데이터
│   ├── train_updated.csv                                         # 전처리된 학습 데이터
│   ├── test.csv                                                  # 원본 테스트 데이터
│   ├── test_updated.csv                                          # 전처리된 테스트 데이터
│   ├── subway_feature.csv                                        # 지하철역 데이터
│   ├── bus_feature.csv                                           # 버스 정류장 데이터
│   └── sample_submission.csv                                     # 샘플 제출 파일
└── docs
├── EDA                                                       # EDA 결과물
│   ├── eda_image_outputs                                     # EDA 시각화 이미지
│   └── 종합적인 탐색적 데이터 분석 (EDA)(For Baseline).pdf      # 종합 EDA 보고서
├── images
│   └── leaderboard.jpg                                       # 리더보드 이미지
├── requirements                                              # 환경 설정 파일
│   ├── requirements_pip_freeze.txt
│   └── environment_py39_rapids_env1.yml
└── team
└── images                                                # 팀원 이미지
├── AI13_홍상호.jpeg
├── AI13_신광명.jpg
├── AI13_최용비.png
├── AI13_이영준.png
└── AI13_김문수.jpg
```

## 3. 데이터 설명

### 데이터셋 개요

이 데이터셋은 다음과 같은 주요 구성 요소가 포함된 한국의 부동산 정보를 포함하고 있습니다:
- 부동산 특성(면적, 층, 건축년도)
- 지리적 좌표
- 대중교통 데이터(지하철역, 버스 정류장)
- 거래 정보
- 타겟 변수: 부동산 가격

총 학습 샘플: 1,118,822
테스트 샘플: 9,272
특성: 53개(숫자형 21개, 범주형 32개)

### EDA

탐색적 데이터 분석을 통해 몇 가지 주요 인사이트를 발견했습니다:

1. **가격 분포**: 타겟 변수는 오른쪽으로 치우친 분포를 보이며, 범위는 ₩350부터 ₩1,450,000까지, 평균 ₩57,992, 중앙값 ₩44,800입니다.

2. **주요 특성 상관관계**:
   - 면적-가격 상관관계: 0.58(강한 양의 관계)
   - 건축년도-가격 상관관계: 0.06(신축 건물일수록 더 비싼 경향)
   - 층-가격 상관관계: 0.15(양의 관계)

3. **이상치 분석**:
   - 가격 이상치: 데이터의 6.8%
   - 면적 이상치: 데이터의 8.0%(21.69㎡에서 122.92㎡)
   - 층 이상치: 데이터의 1.5%(-8층에서 24층)

4. **범주형 특성**:
   - 거래 유형: 직거래 vs. 중개거래, 중개거래는 1.7배 더 높은 가격을 보임
   - 건물 유형: 연립주택이 가장 높은 가격을 보였으며, 도시형 생활주택보다 6.2배 높음

5. **지리적 분석**: 
   - 서울 주택 지역은 구별로 상당한 가격 변동을 보임
   - 교통 연결성이 좋은 지역이 더 높은 가격을 보임

6. **시간적 분석**:
   - 부동산 가격은 기록된 기간(2007-2023) 동안 228.5% 상승
   - 2023년에 가장 높은 평균 가격 관찰됨(₩105,019)
   - 가장 거래가 활발한 달: 6월

### 데이터 처리

1. **결측치 처리**:
   - 90% 이상 결측치가 있는 특성 제거
   - 범주형 변수는 'NULL'로 채움
   - 수치형 변수는 선형으로 보간

2. **특성 공학**:
   - 계약일로부터 연도 및 월 구성요소 생성
   - 강남 지역 표시기 추가(강남여부)
   - 신축 건물 표시기 생성(신축여부, 2015년 이후 건물)
   - 더 나은 분포를 위한 타겟 변수의 로그 변환

3. **좌표 표준화**:
   - 일관된 좌표 규약 보장(X=경도, Y=위도)
   - 공간 데이터셋의 뒤바뀐 좌표 수정

4. **교통 특성 추출**:
   - 가장 가까운 지하철역과 버스 정류장까지의 거리 계산
   - 다양한 반경(500m, 1km, 3km) 내 역/정류장 수 계산
   - 근접성 및 밀도 기반 교통 점수 생성
   - DBSCAN 클러스터링을 사용한 교통 허브 식별
   - 종합적인 접근성 측정을 위한 허브 근접성 특성 생성

## 4. 모델링

### 모델 설명

다음과 같은 이유로 XGBoost를 주요 모델로 선택했습니다:

1. **표 형식 데이터 및 지리적 특성에 대한 강력한 성능**
2. **교통 근접성과 부동산 가격 간의 비선형 관계를 처리하는 능력**
3. **데이터셋에 흔했던 이상치에 대한 견고성**
4. **과적합을 방지하기 위한 조기 중단 지원**
5. **대용량 데이터셋의 빠른 학습을 위한 GPU 가속**

### 모델링 과정

1. **계층적 교차 검증**:
   - 가격 구간을 기반으로 2-폴드 계층적 교차 검증 구현
   - 각 폴드에서 가격 범위의 균형 잡힌 표현 보장

2. **전처리 파이프라인**:
   - 연속 특성의 이상치를 처리하기 위해 RobustScaler 적용
   - 범주형 변수에 LabelEncoder 사용
   - 타겟 변수 로그 변환
   
3. **특성 선택**:
   - SelectFromModel을 사용하여 주요 교통 특성 식별
   - 평균 임계값 이상의 중요도를 가진 특성 유지

4. **XGBoost 구성**:
   ```python
   model = XGBRegressor(
       n_estimators=10000,
       learning_rate=0.05,
       max_depth=6,
       min_child_weight=1,
       subsample=0.8,
       colsample_bytree=0.8,
       early_stopping_rounds=100,
       reg_alpha=0.001,
       reg_lambda=0,
       random_state=42,
       tree_method='gpu_hist'
   )
   ```

5. **성능 최적화**:
   - 병렬화된 교통 특성 추출 구현
   - 효율적인 거리 계산을 위한 공간 인덱스 생성
   - 성능 향상을 위한 벡터화 연산 사용
   - 중복 계산을 피하기 위한 캐싱 구현

### 모델 발전 과정

초기 모델에서 최종 모델까지 다음과 같은 단계로 개선을 진행했습니다:

1. **기본 모델링**: XGBoost와 Random Forest를 기본 파라미터로 학습 (RMSE: ~47000)
2. **파라미터 최적화**: 하이퍼파라미터 튜닝을 통한 성능 개선 (RMSE: ~30000)
3. **교통 특성 추가**: 대중교통 근접성 특성 도입 (RMSE: ~25000)
4. **고급 교통 특성**: 교통 허브 클러스터링 및 근접성 특성 추가 (RMSE: ~22000)
5. **전처리 개선**: RobustScaler 적용 및 이상치 처리 (RMSE: ~19000)
6. **교차 검증 최적화**: 계층적 K-fold 적용 (RMSE: ~18000)
7. **모델 앙상블**: 최적 모델 앙상블 및 최종 튜닝 (RMSE: 12579.01)

각 단계에서 팀원들은 서로 다른 접근 방식을 시도하고 결과를 비교했으며, 가장 효과적인 방법을 최종 모델에 통합했습니다.

## 5. 결과

### 리더보드

<div align="center">
  <img src="https://github.com/AIBootcamp13/upstage-ml-regression-ml_11/blob/wb2x/docs/images/leaderboard.jpg?raw=true" 
       alt="리더보드" 
       width="800" />
</div>

**최종 결과:**
- 순위: 7위
- RMSE: 12579.0120
- 총 제출 횟수: 34
- 팀 이름: ML_11조

**성능 개선 추이:**
- 기본 모델(Baseline): 47133.71 → 최종 모델: 12579.01
- 성능 향상률: 73.3%

### 제출 내역

우리 팀은 다양한 모델과 파라미터를 시도하면서 점진적으로 성능을 향상시켰습니다. 주요 제출 내역은 다음과 같습니다:

| 날짜 | 모델명 | 제출자 | RMSE | 비고 |
|------|--------|--------|------|------|
| 2025.05.15 | XGBoost XY Matched | AI13_이영준 | 19252.93 | 초기 모델 |
| 2025.05.15 | XGBoost Stratified | AI13_이영준 | 18628.73 | 계층적 샘플링 도입 |
| 2025.05.14 | XGBoost Refactor | AI13_이영준 | 18964.85 | 코드 리팩토링 |
| 2025.05.13 | XGB RobustScaler | AI13_홍상호 | 19324.91 | RobustScaler 적용 |
| 2025.05.12 | XGBoost K-fold | AI13_이영준 | 22289.25 | K-fold 교차 검증 |
| 2025.05.10 | XGBoost (Different Features) | AI13_이영준 | 23463.22 | 교통 특성 추가 |
| 2025.05.03 | Random Forest (Tuning) | AI13_이영준 | 23128.38 | 랜덤 포레스트 |
| 2025.05.02 | Baseline Test | AI13_이영준 | 47133.71 | 기본 모델 |

최종적으로 팀의 점수를 12579.01로 향상시켜 7위를 달성했습니다.

### 주요 발견 사항

1. **교통 특성 영향**:
   - 교통 특성은 전체 특성 중요도의 약 15-20%를 차지
   - 지하철역 근접성이 버스 정류장보다 더 강한 영향을 미침
   - 여러 교통 옵션이 있는 교통 허브가 부동산 가치와 가장 높은 상관관계를 보임

2. **특성 중요도 분석**:
   - 주요 교통 특성 포함:
     - nearest_subway_distance(가장 가까운 지하철역 거리)
     - transit_score(교통 점수)
     - subway_stations_1km(1km 내 지하철역 수)
     - hub_proximity_score(허브 근접성 점수)
     - near_public_transit(대중교통 근접 여부)

3. **오류 분석**:
   - 교통 연결성이 좋은 부동산에서 모델 성능이 더 좋음
   - 특이한 특성 조합을 가진 부동산(예: 교통에서 먼 매우 큰 아파트)에서 높은 오류율 관찰
   - 극도로 고가의 부동산에 대한 정밀도 감소

## 6. 향후 작업

1. **고급 허브 클러스터링**:
   - 다양한 밀도에서 개선된 클러스터링을 위한 HDBSCAN 구현
   - 더 정교한 허브 품질 메트릭 생성

2. **노선별 특성**:
   - 지하철 노선별 중요도 가중치 추가(예: 주요 노선 vs. 보조 노선)
   - 환승역 분석 고려

3. **시간적 교통 분석**:
   - 교통 네트워크의 역사적 발전 통합
   - 새 역 개통이 주변 부동산 가치에 미치는 영향 모델링

4. **앙상블 접근 방식**:
   - 다양한 부동산 유형에 대한 특화된 모델 개발
   - 다양한 지역에 대한 위치별 모델 생성

## 참고 자료

- 교통 클러스터링 구현: "도시 이동성 패턴의 공간적 클러스터링" 방법론에서 영감을 받음
- Haversine 거리 계산기: scikit-learn의 구현 기반
- XGBoost 매개변수 튜닝: "Practical XGBoost in Python" 가이드의 모범 사례 따름
- DBSCAN 매개변수 선택: "공간 데이터 클러스터링을 위한 DBSCAN 매개변수 선택" 논문 기반
