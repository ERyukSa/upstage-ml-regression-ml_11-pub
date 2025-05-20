# Transportation Features for Housing Price Prediction: Project Workflow

## Visual Workflow

graph TD
    A[1. Data Loading] --> B[2. Feature Preparation]
    B --> C[2.1 Fix Coordinates]
    C --> D[2.2 Standardize Coordinates]
    D --> E[2.3 Transportation Feature Processing]
    
    E --> E1[2.3.1 Optimize Parallelization]
    E --> E2[2.3.2 Advanced Vectorization]
    E --> E3[2.3.3 Memory Optimization]
    E --> E4[2.3.4 Algorithm Selection]
    
    E1 --> F[2.4 Apply Transportation Processing]
    E2 --> F
    E3 --> F
    E4 --> F
    
    F --> G[2.5 Transportation Feature Integration]
    G --> H[2.6 Save Transportation Features]
    H --> I[2.7 Visualization]
    
    I --> J[3. Feature Engineering]
    J --> K[4. Data Splitting]
    K --> L[5. Preprocessing Pipeline]
    L --> M[6. Model Training]
    M --> N[7. Model Evaluation]
    N --> O[8. Inference on Test Data]
    N --> P[9. Model Optimization]
    P --> Q[10. Feature Importance Visualization]
    O --> R[11. Geographic Visualization]

## 1\. 데이터 로딩

CSV 파일에서 부동산 데이터, 지하철역 데이터, 버스 정류장 데이터를 로드합니다. 데이터에는 부동산 특성, 지리적 좌표 및 교통 정보가 포함됩니다.

## 2\. 특성 준비

누락된 값을 처리하고, 데이터 유형을 변환하며, 연속형 및 범주형 열을 식별하여 원시 데이터를 정리하고 준비합니다.

### 2.1 메인 데이터셋의 좌표 수정

모든 데이터셋의 좌표가 동일한 규칙을 따르도록 합니다. 부동산, 지하철, 버스 데이터 전반에 걸쳐 좌표 시스템을 수정하고 표준화합니다.

### 2.2 좌표 표준화 함수 추가

적절한 공간 계산을 위해 X = 경도(~127) 및 Y = 위도(~37)를 보장하는 좌표 규칙을 표준화하는 함수를 구현합니다.

### 2.3.1 병렬화 최적화 \[벡터화\]

대규모 데이터셋을 효율적으로 처리하기 위한 병렬 처리 프레임워크를 설정합니다. 교통 특성 추출 파이프라인을 조율하는 핵심 함수 process\_transportation\_features를 정의합니다.

### 2.3.2 고급 벡터화

대규모 데이터셋 처리 시 성능을 향상시키기 위해 하버사인 거리와 같은 지리적 계산의 최적화된 벡터화 버전을 구현합니다.

### 2.3.3 메모리 및 I/O 최적화

제한된 RAM으로 대규모 데이터셋을 더 효율적으로 처리하기 위해 데이터 유형 다운캐스팅과 같은 메모리 최적화 기술을 구현합니다.

### 2.3.4 데이터셋 크기 기반 알고리즘 선택

데이터셋의 크기에 따라 최적의 알고리즘과 매개변수를 동적으로 선택하여 다양한 규모에 맞게 처리 전략을 조정합니다.

### 2.4 교통 특성 처리 적용

데이터셋 크기에 따라 병렬 또는 순차 처리를 사용할지 결정하여 최적화된 처리를 적용합니다.

### 2.5 교통 특성 통합

가장 가까운 역까지의 거리, 교통 점수 및 기타 지표를 계산하는 주요 교통 특성 통합 프로세스를 실행합니다.

### 2.6 교통 특성 저장

반복적인 계산을 피하기 위해 교통 특성이 포함된 처리된 데이터를 캐시에 저장합니다.

### 2.7 교통 특성 시각화

지리적 분포, 지하철 근접성이 가격에 미치는 영향, 교통 점수 분포 등 교통 특성의 시각화를 생성합니다.

## 3\. 특성 엔지니어링

기본 및 교통 특성에서 추가 특성을 생성하며, 특히 부동산 특성과 교통 접근성 간의 상호작용 특성을 만듭니다.

## 4\. 데이터 분할

각 폴드에서 대표적인 가격 분포를 보장하기 위해 계층화된 K-폴드를 사용하여 데이터를 훈련 및 검증 세트로 분할합니다.

## 5\. 전처리 파이프라인

특성 스케일링, 범주형 인코딩, 교통 특성 선택 등의 전처리 단계를 구현합니다.

## 6\. 모델 훈련

교차 검증을 사용하여 XGBoost 회귀 모델을 훈련하고 각 폴드의 성능 지표를 추적합니다.

## 7\. 모델 평가

오류 분석, 특성 중요도 및 폴드 간 성능 비교를 포함한 모델의 종합적인 평가를 수행합니다.

### 7.3/7.4 교통 특성 분석

주택 가격 예측에 대한 교통 특성의 영향과 중요성을 구체적으로 분석합니다.

## 8\. 테스트 데이터에 대한 추론

최상의 모델을 테스트 데이터셋에 적용하여 예측을 수행하고 제출 파일을 생성합니다.

## 9\. 교통 특성을 통한 모델 최적화

분석 결과를 바탕으로 교통 특성의 추가 최적화를 위한 권장 사항을 제공합니다.

## 10\. 교통 특성 중요도 시각화

교통 특성 중요도 카테고리 및 순위에 대한 상세한 시각화를 생성합니다.

## 11\. 지리적 가격 예측 시각화

예측된 주택 가격을 지리적으로 시각화하여 가격 패턴을 식별하기 위한 히트맵 및 밀도 플롯을 생성합니다.

## 주요 구성 요소 및 함수

-   **process\_transportation\_features**: 병렬화 및 캐싱을 통해 교통 특성 추출을 조율하는 핵심 함수입니다.
-   **standardize\_coordinates**: 데이터셋 전반에 걸쳐 일관된 좌표 시스템을 보장합니다.
-   **optimize\_spatial\_operations**: 공간 계산에 벡터화 최적화를 적용합니다.
-   **select\_optimal\_algorithms**: 데이터셋 크기에 따라 처리 전략을 동적으로 선택합니다.
-   **create\_distance\_bands**: 분석을 위해 연속적인 거리 측정을 범주형 밴드로 변환합니다.
-   **preprocess\_fold\_data**: 교차 검증 폴드에 대한 특성 전처리를 처리합니다.
-   **select\_important\_transport\_features**: 모델링을 위한 가장 중요한 교통 특성을 선택합니다.

## 향후 작업 권장 사항

-   최적의 거리 임계값을 찾기 위해 다양한 교통 근접성 임계값(예: 800m vs 1km)을 실험합니다.
-   교통 허브를 더 잘 식별하기 위해 다