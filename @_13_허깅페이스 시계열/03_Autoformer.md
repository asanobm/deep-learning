# Autoformer

Autoformer 모델은 Haixu Wu, Jiehui Xu, Jianmin Wang, Mingsheng Long가 제안한 "Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting" 논문에서 소개되었습니다. 이 모델은 트랜스포머를 심층 분해 아키텍처로 확장하여, 예측 과정에서 추세와 계절성 요소를 점진적으로 분해할 수 있습니다.

## Autoformer 모델 개층

Autoformer는 Haixu Wu, Jiehui Xu, Jianmin Wang, Mingsheng Long가 제안한 "Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting" 논문에서 소개된 모델입니다. 이 모델은 트랜스포머를 심층 분해 아키텍처로 확장하여, 예측 과정에서 추세와 계절성 요소를 점진적으로 분해할 수 있습니다. 이는 장기 시계열 예측에 특히 유용하며, 복잡한 시간적 패턴을 효과적으로 처리할 수 있도록 설계되었습니다.

Autoformer는 자기상관 메커니즘을 통해 시계열의 주기성을 기반으로 하위 시계열 수준에서 종속성을 발견하고 표현을 집계합니다. 이는 기존의 셀프 어텐션 메커니즘을 능가하는 효율성과 정확성을 제공합니다. Autoformer는 에너지, 교통, 경제, 날씨, 질병 등 다양한 분야에서 뛰어난 성능을 보이며, 여러 벤치마크에서 최첨단 정확도를 달성했습니다.

## 주요 구성 요소

- **AutoformerConfig**: 모델의 설정을 정의하는 클래스입니다. 예측 길이, 컨텍스트 길이, 분포 출력, 손실 함수, 입력 크기, 래그 시퀀스, 스케일링, 시간 특징 수, 동적 및 정적 특징 수, 카디널리티, 임베딩 차원, 모델 차원, 인코더 및 디코더의 어텐션 헤드 수, 레이어 수, FFN 차원, 활성화 함수 등을 설정할 수 있습니다.

- **AutoformerForPrediction**: 예측을 위한 모델 클래스입니다. 과�� 값과 미래 값을 입력으로 받아 예측을 수행하며, 추가적인 특징을 활용할 수 있습니다.

이 모델은 시계열 데이터의 복잡한 패턴을 효과적으로 처리할 수 있도록 설계되어 있으며, 다양한 실제 응용 분야에서 활용될 수 있습니다.

## AutoformerConfig 클래스

### Parameters

* `prediction_length`: (int), **예측 길이** 모델이 얼마나 멀리 미래를 예측할 지를 정의합니다.
* `context_length`: (int, optional, defaults to `prediction_length`), **컨텍스트 길이** 모델에게 더 많은 과거 정보를 제공하여 더 나은 예측을 할 수 있도록 도와줍니다. 데이터 세트의 특성에 따라 이를 조정하는 것이 성능 향상에 도움이 될 수 있습니다.
* `distribution_output`: (string, optional, defaults to "student_t"), **분포 출력** 올바른 분포를 선택하는 것("student_t", "normal", "negative_binomial")은 모델이 예측의 불확실성과 꼬리를 추정하는 방식에 영향을 줍니다.
* `loss`: (string, optional, defaults to "nll"), **손실 함수** 현재는 음의 로그 우도("negative log likelihood")만 지원합니다.
* `input_size`: (int, optional, defaults to 1), **입력 크기** 단항 예측(univariate)에서 중요합니다. 모델이 고려하는 타겟 변수의 수에 직접 영향을 미칩니다.
* `lags_sequence`: (list[int], optional, defaults to [1, 2, 3, 4, 5, 6, 7]), **래그 시퀀스** 모델이 과거 값을 참조하는 시점을 정의합니다.
* `scaling`: (bool, optional, defaults to True), **스케일링** 입력 특징을 적절히 스케일링(mean, standard deviation)하는 것은 모델 수렴 및 성능에 상당한 영향을 줍니다.
* `num_time_features`: (int, optional, defaults to 0), **시간 특징 수** 모델이 고려하는 시간적 특성의 수를 정의합니다.
* `num_dynamic_real_features`: (int, optional, defaults to 0), **동적 실수 특징 수** 모델이 고려하는 동적 실수 특징의 수를 정의합니다.
* `num_static_categorical_features`: (int, optional, defaults to 0), **정적 범주형 특징 수** 모델이 고려하는 정적 범주형 특징의 수를 정의합니다.
* `num_static_real_features`: (int, optional, defaults to 0), **정적 실수 특징 수** 모델이 고려하는 정적 실수 특징의 수를 정의합니다.
* `cardinality`: (list[int], optional), **카디널리티** 각 범주형 특징의 카디널리티를 정의합니다.
* `embedding_dimension`: (list[int], optional), **임베딩 차원** 각 범주형 특징의 임베딩 차원을 정의합니다.
* `d_model`: (int, optional, defaults to 64), **모델 차원** 트랜스포머 모델의 차원을 정의합니다.
* `encoder_attention_heads`: (int, optional, defaults to 2), **인코더 어텐션 헤드 수** 인코더의 어텐션 헤드 수를 정의합니다.
* `decoder_attention_heads`: (int, optional, defaults to 2), **디코더 어텐션 헤드 수** 디코더의 어텐션 헤드 수를 정의합니다.
* `encoder_layers`: (int, optional, defaults to 2), **인코더 레이어 수** 인코더의 레이어 수를 정의합니다.
* `decoder_layers`: (int, optional, defaults to 2), **디코더 레이어 수** 디코더의 레이어 수를 정의합니다.
* `encoder_ffn_dim`: (int, optional, defaults to 32), **인코더 FFN 차원** 인코더의 피드 포워드 네트워크 차원을 정의합니다.
* `decoder_ffn_dim`: (int, optional, defaults to 32), **디코더 FFN 차원** 디코더의 피드 포워드 네트워크 차원을 정의합니다.
* `activation_function`: (string, optional, defaults to "gelu"), **활성화 함수** 모델의 활성화 함수를 정의합니다.

