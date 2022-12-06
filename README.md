# AI-X-DeepLearning

### Title: Mobile Price Classification

### Members: 
          이상빈, 건설환경공학과, lsb10a@naver.com
          최영우, 기계공학부, woo970908@naver.com
          황환이, 기계공학부, asa517@hanyang.ac.kr


## Index
####           I. Proposal
####           II. Datasets
####           III. Methodology
####           IV. Evaluation & Analysis
####           V. Related Works
####           VI. Conclusion: Discussion

## I. Proposal
+ Motivation: 

   AI를 처음 다루어보는 팀원들이 모여 예측을 하는 Deep Learning Model을 
  만들어 보고자 기본적인 Classification을 경험해보고자 했습니다.Classification에는 여러 기법들이 있지만 그 중 저희 팀은 Decision tree와 Random forest 알고리즘에 대해
  학습하였고 어느 알고리즘의 성능이 더 좋은 지 확인해보기 위해 하나의 예제를 통해 비교해보고자 하였습니다.또, n_estimator, max_depth, feature 갯수 등에 따라 알고리즘의 성능이
  바뀐다는 것을 알게 되었고 실제 '기기 스펙을 통한 모바일 기기 가격 예측' 이라는 예제를 통하여 이를 보여주고자 하였습니다. 

+ What do you want to see at the end?

  Decision tree와 Random forest를 정확도로 비교하여 어느 알고리즘의 성능이 더 좋은지 보여줄 것입니다. 또, 반복문을 통한 비교를 통해 Estimator, max_depth 등의 최적의 변수를 찾아 
  이를 적용시켜 학습 시킬 예정이며 최종적으로 모바일 기기의 여러 스펙 중 feature importance가 높은 feature를 추출하여 모바일 기기의 가격을 예측하는 분류 학습을 진행할 예정입니다.
  최종적으로 소비자가 모바일 기기를 구매할 때, 가격을 예측하기 위해 어느 기능을 고려해야 합리적인 가격 예측에 도움이 되는 지에 대한 기준을 정립할 계획입니다.
## II. Datasets
+ https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification
+ 학습 데이터는 kaggle의 Datasets을 사용할 예정이며 두 개의 csv 파일이 주어지지만, test.csv에는 정답값이 정해져 있지 않아 train.csv만 사용할 예정입니다. 각 csv 파일은 20개의 Column으로 이루어져 있습니다. 색인을 포함하여 총 21개의 Column을 가지게 됩니다. 각각의 Column은 다음과 같습니다.

          battery_power     : 배터리가 한번에 저장할 수 있는 총에너지 양 (mAh)
          blue              : Bluetooth 기능의 제공여부 (0 or 1)
          clock_speed       : Microprocessor가 명령을 실행하는 속도
          dual_sim          : Dual Sim 기능의 제공여부 (0 or 1)
          fc                : 전면 카메라의 성능 (Mega Pixel)
          four_g            : 4G 통신의 제공여부 (0 or 1)
          int_memory        : 내장 메모리의 용량 (GB)
          m_dep             : 모바일 기기의 두께 (cm)
          mobile_wt         : 모바일 기기의 무게 (g)
          n_cores           : 프로세서의 코어 개수
          pc                : 후면 카메라의 성능 (Mega Pixel)
          px_height         : 높이 해상도 (Pixel)
          px_width          : 폭 해상도 (Pixel)
          ram               : RAM 성능 (MB)
          sc_h              : 스크린 높이 (cm)
          sc_w              : 스크린 폭 (cm)
          talk_time         : 배터리 최대 지속시간
          three_g           : 3G 통신의 제공여부 (0 or 1)
          touch_screen      : 터치 스크린 제공여부 (0 or 1)
          wifi              : WIFI 기능 제공여부 (0 or 1)
          price_range       : 해당 모바일 기기의 가격이 속한 범위

<img src="https://user-images.githubusercontent.com/91457152/204254565-88a3eca5-a554-4234-a0a3-01189147e02b.png"  width="500" height="300"/><img src="https://user-images.githubusercontent.com/91457152/204254747-8b28b429-51dc-4a12-a43b-0dfa68ad1d32.png"  width="500" height="300"/><img src="https://user-images.githubusercontent.com/91457152/204254754-a13768fe-4b87-465d-a703-065e8e3157de.png"  width="500" height="300"/><img src="https://user-images.githubusercontent.com/91457152/204254763-36faeb65-7615-415f-b2cc-7657fef26c22.png"  width="500" height="300"/><img src="https://user-images.githubusercontent.com/91457152/204254765-5f428ad8-6918-4e18-b06a-57e52442abf9.png"  width="500" height="300"/><img src="https://user-images.githubusercontent.com/91457152/204254767-60e7d04c-eea4-48dc-ae8f-4351d54119cb.png"  width="500" height="300"/>


## III. Methodology
+ Explaining your choice of algorithms (methods)
+ Explaining features (if any)
모델 소개

로지스틱 회귀, 커널 기법, SVM(Support Vector Machine) 등 다양한 분류 기법이 있습니다.
저희는 그 중에서 계속 질문을 해가면서 주어진 데이터셋을 분류하는 방식인 Decision Tree라는 기법을 알아보았습니다. 사람의 추론 방식을 모방한 기법으로 기존 기법들과 달리, 계속 질문을 해가면서 데이터셋을 나누는 방식입니다. 

![image](https://user-images.githubusercontent.com/119299773/204279084-b81cb441-6b86-4ab6-a562-939cf66ea8db.png)


전체적으로 보면 나무를 뒤집어놓은 것과 같은 모양입니다. 아시다시피 초기지점은 root node이고 분기가 거듭될 수록 그에 해당하는 데이터의 개수는 줄어듭니다. 각 terminal node에 속하는 데이터의 개수를 합하면 root node의 데이터수와 일치합니다. 바꿔 말하면 terminal node 간 교집합이 없다는 뜻입니다. 한편 terminal node의 개수가 분리된 집합의 개수입니다. 예컨대 위 그림처럼 terminal node가 3개라면 전체 데이터가 3개의 부분집합으로 나눠진 셈입니다.

의사결정나무는 분류(classification)와 회귀(regression) 모두 가능합니다. 범주나 연속형 수치 모두 예측할 수 있다는 말입니다. 의사결정나무의 범주예측, 즉 분류 과정은 이렇습니다. 새로운 데이터가 특정 terminal node에 속한다는 정보를 확인한 뒤 해당 terminal node에서 가장 빈도가 높은 범주에 새로운 데이터를 분류하게 됩니다. 

회귀의 경우 해당 terminal node의 종속변수(y)의 평균을 예측값으로 반환하게 되는데요, 이 때 예측값의 종류는 terminal node 개수와 일치합니다. 만약 terminal node 수가 3개뿐이라면 새로운 데이터가 100개, 아니 1000개가 주어진다고 해도 의사결정나무는 딱 3종류의 답만을 출력하게 될 겁니다.

그렇다면 데이터를 분할한다는 건 정확히 어떤 의미를 지니는걸까요? 설명변수(X)가 3개짜리인 다변량 데이터에 의사결정나무를 적용한다고 가정하고 아래 그림을 보시면 좋을 것 같습니다.

![image](https://user-images.githubusercontent.com/119299773/204279492-fd3eac37-e335-41cd-a905-955e6fc1e0c3.png)

아무런 분기가 일어나지 않은 상태의 root node는 A입니다. 변수가 3개짜리이니 왼쪽 그래프처럼 3차원 공간에 있는 직육면체를 A라고 상정해도 좋을 것 같네요. A가 어떤 기준에 의해 B와 C로 분할됐다고 생각해 봅시다. 그렇다면 두번째 그림처럼 A가 두 개의 부분집합으로 나뉘었다고 상상해 볼 수 있겠습니다. 마지막으로 B가 D와 C로 분할됐다면, 3번째 줄의 그림처럼 될 겁니다. 이 예시에서 terminal node는 C, D, E 세 개 인데요, 이를 데이터 공간과 연관지어 생각해보면 전체 데이터 A가 세 개의 부분집합으로 분할된 것 또한 알 수 있습니다. D 특성을 갖고 있는 새로운 데이터가 주어졌을 때 의사결정나무는 D 집합을 대표할 수 있는 값(분류=최빈값, 회귀=평균)을 반환하는 방식으로 예측합니다.

불순도/불확실성

의사결정나무는 한번 분기 때마다 변수 영역을 두 개로 구분하는 모델이라고 설명을 드렸는데요, 그렇다면 대체 어떤 기준으로 영역을 나누는 걸까요? 이 글에서는 타겟변수(Y)가 범주형 변수인 분류나무를 기준으로 설명하겠습니다.

결론부터 말씀드리면 분류나무는 구분 뒤 각 영역의 순도(homogeneity)가 증가, 불순도(impurity) 혹은 불확실성(uncertainty)이 최대한 감소하도록 하는 방향으로 학습을 진행합니다. 순도가 증가/불확실성이 감소하는 걸 두고 정보이론에서는 정보획득(information gain)이라고 합니다. 이번 챕터에서는 어떤 데이터가 균일한 정도를 나타내는 지표, 즉 순도를 계산하는 3가지 방식에 대해 살펴보겠습니다. 우선 그림을 보시죠.

![image](https://user-images.githubusercontent.com/119299773/204279827-93b6d6ec-9e2b-47fc-a28e-2086502fdd75.png)

먼저 설명드릴 지표는 엔트로피(entropy)입니다. m개의 레코드가 속하는 A영역에 대한 엔트로피는 아래와 같은 식으로 정의됩니다. (Pk=A영역에 속하는 레코드 가운데 k 범주에 속하는 레코드의 비율)

![image](https://user-images.githubusercontent.com/119299773/204279937-51c1a874-244e-4a8c-8a09-53b2073c7168.png)

이 식을 바탕으로 오렌지색 박스로 둘러쌓인 A 영역의 엔트로피를 구해보겠습니다. 전체 16개(m=16) 가운데 빨간색 동그라미(범주=1)는 10개, 파란색(범주=2)은 6개이군요. 그럼 A 영역의 엔트로피는 다음과 같습니다.

![image](https://user-images.githubusercontent.com/119299773/204280025-4c9a9f56-0ea7-49bd-b04f-99847ac9ad8e.png)

여기서 A 영역에 빨간색 점선을 그어 두 개 부분집합(R1, R2)으로 분할한다고 가정해 봅시다. 두 개 이상 영역에 대한 엔트로피 공식은 아래 식과 같습니다. 이 공식에 의해 분할 수 A 영역의 엔트로피를 아래와 같이 각각 구할 수 있습니다. (Ri=분할 전 레코드 가운데 분할 후 i 영역에 속하는 레코드의 비율)

![image](https://user-images.githubusercontent.com/119299773/204280299-9dc6e0d8-a063-4372-9b1b-4e4e9a17cfe2.png)

여기서 A 영역에 빨간색 점선을 그어 두 개 부분집합(R1, R2)으로 분할한다고 가정해 봅시다. 두 개 이상 영역에 대한 엔트로피 공식은 아래 식과 같습니다. 이 공식에 의해 분할 수 A 영역의 엔트로피를 아래와 같이 각각 구할 수 있습니다. (Ri=분할 전 레코드 가운데 분할 후 i 영역에 속하는 레코드의 비율)

![image](https://user-images.githubusercontent.com/119299773/204280429-3eb21a7d-2db4-41d8-acc0-34dd06dc6677.png)

그럼 분기 전과 분기 후의 엔트로피가 어떻게 변화했는지 볼까요? 분기 전 엔트로피가 0.95였는데 분할한 뒤에 0.75가 됐군요. 0.2만큼 엔트로피 감소(=불확실성 감소=순도 증가=정보획득)한 걸로 봐서 의사결정나무 모델은 분할한 것이 분할 전보다 낫다는 판단 하에 데이터를 두 개의 부분집합으로 나누게 됩니다. 다시 한번 말씀드리지만 의사결정나무는 구분 뒤 각 영역의 순도(homogeneity)가 증가/불확실성(엔트로피)가 최대한 감소하도록 하는 방향으로 학습을 진행합니다.

순도와 관련해 부연설명을 드리면 A 영역에 속한 모든 레코드가 동일한 범주에 속할 경우(=불확실성 최소=순도 최대) 엔트로피는 0입니다. 반대로 범주가 둘뿐이고 해당 개체의 수가 동일하게 반반씩 섞여 있을 경우(=불확실성 최대=순도 최소) 엔트로피는 1의 값을 갖습니다. 엔트로피 외에 불순도 지표로 많이 쓰이는 지니계수(Gini Index) 공식은 아래와 같습니다.

![image](https://user-images.githubusercontent.com/119299773/204280526-4802a02d-6da3-46ea-b13a-6539e3e58ce3.png)

아래는 범주가 두 개일 때 한쪽 범주에 속한 비율(p)에 따른 불순도의 변화량을 그래프로 나타낸 것입니다. 보시다시피 그 비율이 0.5(두 범주가 각각 반반씩 섞여 있는 경우)일 때 불순도가 최대임을 알 수가 있습니다. 오분류오차(misclassification error)는 따로 설명드리지 않은 지표인데요, 오분류오차는 엔트로피나 지니계수와 더불어 불순도를 측정할 수 있긴 하나 나머지 두 지표와 달리 미분이 불가능한 점 때문에 자주 쓰이지는 않는다고 합니다.

![image](https://user-images.githubusercontent.com/119299773/204280592-8191d4d9-ba9e-42fc-aa0f-d17daf5183a4.png)

이상으로 의사결정나무에 대해 살펴보았습니다. 의사결정나무는 계산복잡성 대비 높은 예측 성능을 내는 것으로 정평이 나 있습니다. 아울러 변수 단위로 설명력을 지닌다는 강점을 가지고 있습니다. 다만 의사결정나무는 결정경계(decision boundary)가 데이터 축에 수직이어서 특정 데이터에만 잘 작동할 가능성이 높습니다.

이같은 문제를 극복하기 위해 등장한 모델이 바로 랜덤포레스트인데요, 같은 데이터에 대해 의사결정나무를 여러 개 만들어 그 결과를 종합해 예측 성능을 높이는 기법입니다. 최종적으로 저희는 결정나무기법을 바탕으로 랜덤포레스트를 사용하기로 하였습니다.

랜덤 포레스트는 지도 머신 러닝 알고리즘입니다. 정확성, 단순성 및 유연성으로 인해 가장 많이 사용되는 알고리즘 중 하나입니다. 분류 및 회귀 작업에 사용할 수 있다는 사실과 비선형 특성을 결합하면 다양한 데이터 및 상황에 매우 적합합니다.

![image](https://user-images.githubusercontent.com/119299773/204281590-fedaa99d-ce37-4964-9006-fd717a0d51a2.png)

"랜덤 의사결정 포레스트"라는 용어는 1995년 Tin Kam Ho에 의해 처음 제안되었습니다. Ho는 예측을 생성하기 위해 랜덤 데이터를 사용하는 공식을 개발했습니다. 그후 2006년에 Leo Breiman과 Adele Cutler가 알고리즘을 확장하여 오늘날 우리가 알고 있는 랜덤 포레스트를 만들었습니다. 이것은 이 기술과 그 이면의 수학과 과학이 여전히 비교적 새로운 것임을 의미합니다.

의사 결정 트리의 숲이 자라기 때문에 "포레스트"라고 합니다. 그런 다음 이 트리의 데이터를 병합하여 가장 정확한 예측을 보장합니다. 단독 의사 결정 트리는 하나의 결과와 좁은 범위의 그룹을 갖지만, 포레스트는 더 많은 수의 그룹 및 결정으로 보다 정확한 결과를 보장합니다. 랜덤 기능 하위 집합 중에서 최상의 기능을 찾아 모델에 임의성을 추가하는 추가 이점이 있습니다. 전반적으로 볼 때 이러한 이점으로 하여 많은 데이터 사이언티스트가 선호하는 광범위한 다양성을 가진 모델이 생성됩니다.

의사 결정 트리와 랜덤 포레스트의 차이점

랜덤 포레스트는 의사 결정 트리의 그룹입니다. 그러나 양자 사이에는 몇 가지 차이점이 있습니다. 의사 결정 트리는 의사 결정에 사용하는 규칙을 만드는 경향이 있습니다. 랜덤 포레스트는 기능을 무작위로 선택하고 관찰하여 의사 결정 트리의 포리스트를 만든 다음 결과를 평균화합니다.

이론에 따르면 많은 수의 상관되지 않은 트리가 하나의 개별 의사 결정 트리보다 더 정확한 예측을 생성합니다. 이는 많은 트리들이 함께 작동하여 개별 오류와 과적합으로부터 서로를 보호하기 때문입니다.

랜덤 포레스트가 제대로 작동하려면 다음 세 가지가 필요합니다.

● 모델이 추측만 하지 않도록 신호가 식별가능해야 합니다.
● 트리로 만든 예측은 다른 트리들과 상관 관계가 작아야 합니다.
● 어느 정도 예측력이 있는 기능에 대해 GI=GO여야 합니다.

랜덤 포레스트의 이점

상대적 중요성을 측정하기 쉬움
해당 기능을 사용하여 해당 포레스트에 있는 모든 트리의 불순물을 줄이는 노드를 보면 기능의 중요성을 쉽게 측정할 수 있습니다. 변수를 치환하기 전과 후의 차이를 쉽게 볼 수 있으며, 이는 해당 변수의 중요도를 측정합니다.

다재다능
랜덤 포레스트는 분류 및 회귀 작업 모두에 사용할 수 있기 때문에 매우 다재다능합니다. 변환이나 재조정 없이 이진 및 숫자 기능과 범주형 기능을 쉽게 처리할 수 있습니다. 거의 모든 다른 모델과 달리 모든 유형의 데이터에서 매우 효율적입니다.

과적합 없음
포레스트에 트리가 충분하면 과적합의 위험이 거의 또는 전혀 없습니다. 의사 결정 트리도 과적합으로 끝날 수 있습니다. 랜덤 포레스트는 하위 집합에서 다양한 크기의 트리를 만들고 결과를 결합하여 이를 방지합니다.

높은 정확도
하위 그룹 간에 상당한 차이가 있는 여러 트리를 사용하면 랜덤 포레스트가 매우 정확한 예측 도구가 됩니다.

데이터 관리에 소요되는 시간 단축
기존 데이터 처리에서는 귀중한 시간의 상당 부분이 데이터를 정리하는 데 사용됩니다. 랜덤 포레스트는 누락된 데이터를 잘 처리하므로 이를 최소화합니다. 완전한 데이터와 불완전한 데이터의 예측을 비교한 테스트에 따르면 성능이 거의 동일했습니다. 이상 데이터 및 비선형 기능은 기본적으로 삭제됩니다.

랜덤 포레스트 기술은 또한 모집단 및 기타 불균형 데이터 세트의 오류 균형을 맞추는 데 사용됩니다. 오류율을 최소화하여 이를 수행하므로 더 큰 클래스는 더 낮은 오류율을 가지며 더 작은 클래스는 더 큰 오류율을 갖습니다.

빠른 학습 속도
랜덤 포레스트는 하위 집합 기능을 사용하기 때문에 수백 가지의 다양한 기능을 빠르게 평가할 수 있습니다. 즉 생성된 포레스트를 저장하고 향후 재사용할 수 있기 때문에 예측 속도도 다른 모델보다 빠르게 됩니다.
## IV. Evaluation & Analysis
+ Graphs, tables, any statistics (if any)

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
```

```python
train = pd.read_csv("train.csv")
train.head()
```
	battery_power	blue	clock_speed	dual_sim	fc	four_g	int_memory	m_dep	mobile_wt	n_cores	...	px_height	px_width	ram	sc_h	sc_w	talk_time	three_g	touch_screen	wifi	price_range
0	842	0	2.2	0	1	0	7	0.6	188	2	...	20	756	2549	9	7	19	0	0	1	1
1	1021	1	0.5	1	0	1	53	0.7	136	3	...	905	1988	2631	17	3	7	1	1	0	2
2	563	1	0.5	1	2	1	41	0.9	145	5	...	1263	1716	2603	11	2	9	1	1	0	2
3	615	1	2.5	0	0	0	10	0.8	131	6	...	1216	1786	2769	16	8	11	1	0	0	2
4	1821	1	1.2	0	13	1	44	0.6	141	2	...	1208	1212	1411	8	2	15	1	1	0	1
5 rows × 21 columns

```python
train.info()
```
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2000 entries, 0 to 1999
Data columns (total 21 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   battery_power  2000 non-null   int64  
 1   blue           2000 non-null   int64  
 2   clock_speed    2000 non-null   float64
 3   dual_sim       2000 non-null   int64  
 4   fc             2000 non-null   int64  
 5   four_g         2000 non-null   int64  
 6   int_memory     2000 non-null   int64  
 7   m_dep          2000 non-null   float64
 8   mobile_wt      2000 non-null   int64  
 9   n_cores        2000 non-null   int64  
 10  pc             2000 non-null   int64  
 11  px_height      2000 non-null   int64  
 12  px_width       2000 non-null   int64  
 13  ram            2000 non-null   int64  
 14  sc_h           2000 non-null   int64  
 15  sc_w           2000 non-null   int64  
 16  talk_time      2000 non-null   int64  
 17  three_g        2000 non-null   int64  
 18  touch_screen   2000 non-null   int64  
 19  wifi           2000 non-null   int64  
 20  price_range    2000 non-null   int64  
dtypes: float64(2), int64(19)
memory usage: 328.2 KB
```

```python
origin_feature = train.drop('price_range')
target = train['price_range']

x_train, x_test, y_train, y_test = train_test_split(origin_feature, target, test_size = 0.3)
```

```python
params = {
    'max_depth': [8, 10, 12, 16],
    'n_estimators': [250, 500, 750, 1000],
    'min_samples_split': [8, 12]
}

rf_model = RandomForestClassifier(random_state = 13, n_jobs = -1)
grid_cv = GridSearchCV(rf_model, param_grid = params, cv = 2, n_jobs = -1)
grid_cv.fit(x_train, y_train)
```

```python
cv_results = pd.DataFrame(grid_cv.cv_results_)
cv_results.columns
```
```
Index(['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time',
       'param_max_depth', 'param_min_samples_split', 'param_n_estimators',
       'params', 'split0_test_score', 'split1_test_score', 'mean_test_score',
       'std_test_score', 'rank_test_score'],
      dtype='object')
```
```python
target_col = ['rank_test_score', 'mean_test_score', 'param_n_estimators', 'param_max_depth']
cv_results[target_col].sort_values('rank_test_score').head()
```

|index|rank_test_score|mean_test_score|param_n_estimators|param_max_depth|
|---|---|---|---|---|
|11|1|0.856429|1000|10|
|2|2|0.854286|750|8|
|19|2|0.854286|1000|12|
|9|2|0.854286|500|10|
|3|5|0.853571|1000|8|

```python
print(grid_cv.best_params_)
print(grid_cv.best_score_)
```
```
{'max_depth': 10, 'min_samples_split': 8, 'n_estimators': 1000}
0.8564285714285714
```

```python
best_data = grid_cv.best_estimator_
best_data.fit(x_train, y_train.values.reshape(-1, ))

pred1 = best_data.predict(x_test)

accuracy_score(y_test, pred1)
```
```
0.87
```

```python
best_cols_values = best_data.feature_importances_
best_cols = pd.Series(best_cols_values, index = x_train.columns)
top8_cols = best_cols.sort_values(ascending=False)[:8]
top8_cols
```
```
ram              0.555619
battery_power    0.075832
px_width         0.053679
px_height        0.052308
mobile_wt        0.034910
int_memory       0.032589
talk_time        0.025553
sc_w             0.022609
dtype: float64
```

```python
plt.figure(figsize = (8, 8))
sns.barplot(x = top8_cols, y = top8_cols.index)
plt.show()
```
![image](https://user-images.githubusercontent.com/91457152/205852440-06eca3f5-2ff4-452b-a656-79a2f845153a.png)

```python
x_train_re = x_train[top8_cols.index]
x_test_re = x_test[top8_cols.index]

rf_model_re = grid_cv.best_estimator_
rf_model_re.fit(x_train_re, y_train.values.reshape(-1, ))

pred1_re = rf_model_re.predict(x_test_re)

accuracy_score(y_test, pred1_re)
```
```
0.88
```
```python
conf_matrix = pd.DataFrame(confusion_matrix(y_test, pred1_re))
sns.heatmap(conf_matrix, annot = True, fmt = 'd', linewidths = 2, cmap = 'Blues')
```
![image](https://user-images.githubusercontent.com/91457152/205853406-58c37ca0-7024-4f7d-91d7-2cbea5d77208.png)

```python
top6_cols = best_cols.sort_values(ascending=False)[:6]
top4_cols = best_cols.sort_values(ascending=False)[:4]
top3_cols = best_cols.sort_values(ascending=False)[:3]
```
```python
x_train_re = x_train[top6_cols.index]
x_test_re = x_test[top6_cols.index]

rf_model_re = grid_cv.best_estimator_
rf_model_re.fit(x_train_re, y_train.values.reshape(-1, ))

pred1_re = rf_model_re.predict(x_test_re)

accuracy_score(y_test, pred1_re)

x_train_re = x_train[top4_cols.index]
x_test_re = x_test[top4_cols.index]

rf_model_re = grid_cv.best_estimator_
rf_model_re.fit(x_train_re, y_train.values.reshape(-1, ))

pred1_re = rf_model_re.predict(x_test_re)

accuracy_score(y_test, pred1_re)

x_train_re = x_train[top3_cols.index]
x_test_re = x_test[top3_cols.index]

rf_model_re = grid_cv.best_estimator_
rf_model_re.fit(x_train_re, y_train.values.reshape(-1, ))

pred1_re = rf_model_re.predict(x_test_re)

accuracy_score(y_test, pred1_re)
```
```
0.8983333333333333
0.905
0.865


## V. Related Work (e.g., existing studies)
+ Tools, libraries, blogs, or any documentation that you have used to do this project.
https://ratsgo.github.io/machine%20learning/2017/03/26/tree/
https://www.tibco.com/ko/reference-center/what-is-a-random-forest

+ Dataset: kaggle, https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification
+ Theory: 
+ Codes: AI+X: Machine Learning

## VI. Conclusion: Discussion
