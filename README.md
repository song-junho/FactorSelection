# FactorSelection

Target: Asset 및 RiskFactor Selection 데이터 생성  
세부 팩터별로 5분위 값의 데이터를 생성

향후 데이터 결합을 통한 새로운 팩터 가공을 위해 원데이터를 Z_Scroe로 변환하여 저장한다.

***
### 1. Asset
#### 1) Stock
1. Value  
  a. 영업이익 멀티플 (ttm)  
  b. 영업이익 멀티플 3년 평균  
  c. a & b 값의 스프레드  
  . OutPut => stock_factor_value_quantiling.pickle
  
2. Growth  
  a. 분기 영업이익 성장률 (qoq, yoy)  
  b. 분기 영업이익 성장률  3년 평균 (qoq, yoy)  
  c. a & b 값의 스프레드  (qoq, yoy)  
  . OutPut => stock_factor_factor_quantiling.pickle
  
3. Momentumn
4. Size


#### 2) Bond
#### 3) Commodity

