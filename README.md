# week1
<img width="1500" height="900" alt="image" src="https://github.com/user-attachments/assets/882c620d-922f-4a7b-b844-196fef19daf4" />

# 南瓜初版  
<img width="1800" height="1200" alt="image" src="https://github.com/user-attachments/assets/7c6d4029-6df0-4eab-92a0-ed59e974658d" />  
<img width="1800" height="900" alt="image" src="https://github.com/user-attachments/assets/21de61fc-bed3-4aa6-a28f-e78b818b2519" />

# 南瓜优化版
## 南瓜价格预测模型深度分析
### 数据细节分析  
- 日期信息价值分析  
- 日期信息在转换为年、月、日等特征后，通过特征重要性分析显示其重要性排名较低  
- 价格随时间变化的趋势图中，季节性波动不明显，说明日期对价格预测贡献有限  
- 删除日期相关特征后，模型性能下降幅度较小（约2-3%的R²下降）  
### 样本分布情况
- ​​品种分布​​  
"HOWDEN"类型占比约35%（样本充分）  
"PIE TYPE"占比约25%  
其他特殊品种（如"FAIRY TALE"）合计占比<5%（样本不足）  
- ​​规格分布​   
"MEDIUM"规格占比约45%  
"LARGE"规格占比约30%  
其他规格（如"JUMBO"）样本较少  
- ​​城市分布​​  
芝加哥、纽约样本占比最高（各约20%）  
中小城市样本分散，单个城市占比<5%  
- ​​产地分布​​
伊利诺伊州产地占比约40%（样本充分）  
加利福尼亚州占比约25%  
其他州样本较少且分散  
### 特征处理细节
### 特征相关性分析  
​​- 热力图观察​​  
价格区间(Price_Range)与平均价格(Avg_Price)存在中等相关性(r≈0.4)  
月份与日度特征存在天然相关性（需注意但不必须处理）  
- ​​多重共线性处理​​
树模型对多重共线性不敏感，线性模型需注意  
实际测试显示删除共线性特征对树模型性能影响<1%  
### 离散值编码影响  
​​顺序编码测试结果​​  
线性模型：R²下降约15%（对编码顺序敏感）  
树模型：R²下降约3%（相对稳健但仍受影响）  
最佳实践：始终使用One-Hot编码   
### 时间特征重要性  
​​删除测试​​  
保留完整时间特征：R²=0.82  
仅保留月份：R²=0.80  
完全删除时间特征：R²=0.78  
结论：时间特征提供约4%的性能提升  
### 模型细节分析  
- 树模型关键参数  
​​最大深度(max_depth)​​：  
深度=3：欠拟合（训练集R²=0.65）  
深度=7：最佳平衡（测试集R²=0.82）  
深度=None：过拟合（训练-测试差距>15%）  
​​叶子节点最小样本数(min_samples_leaf)​​：  
值=1：易过拟合  
值=5：最佳实践  
值>10：欠拟合风险  
​​树数量(n_estimators)​​：  
随机森林在n=200时达到性能平台  
XGBoost在n=150后收益递减  
3.2 模型学到的"智能"  

品种优先级：HOWDEN > PIE TYPE > 其他（价格递减）
模型准确捕捉市场对不同品种的偏好差异

规格影响非线性：MEDIUM规格价格峰值，JUMBO规格价格反降
反映运输成本和市场需求平衡

产地溢价：伊利诺伊州本地供应价格低于外州供应
体现运输成本对价格的影响

时间效应：9-10月价格谷底（收获季供应充足）
模型自主发现季节性规律
### 模型对比洞见
​​XGBoost表现最佳​​：  
测试集R²=0.85  
关键优势：有效处理混合特征类型  
​​随机森林稳健性​​：  
交叉验证标准差最小（±0.02）  
​​LightGBM训练效率​​：  
训练时间仅为其他模型的1/3  
### 建议
​​- 数据收集重点​​  
加强稀缺品种(如FAIRY TALE)的数据收集  
增加中小城市的样本覆盖  
​​- 特征工程优化​​  
保留完整时间特征但降低其权重  
对高基数分类特征采用目标编码  
- ​​模型选择策略​​
首选XGBoost/LightGBM  
次选随机森林（当需要更强解释性时）  
设置合理的树深度限制（5-7层）
### 结果  
<img width="1800" height="900" alt="image" src="https://github.com/user-attachments/assets/d107a45f-25c5-4557-a34e-bec0412e9c53" />
<img width="1800" height="900" alt="image" src="https://github.com/user-attachments/assets/679bcba3-0009-4b5c-8ef2-50281e4f04ea" />
<img width="1800" height="900" alt="image" src="https://github.com/user-attachments/assets/a519d95e-8f13-4b63-adfd-928d92b081c5" />
<img width="1800" height="900" alt="image" src="https://github.com/user-attachments/assets/20b8c5dc-e19d-484f-80e8-22477a7fdae4" />
<img width="1800" height="900" alt="image" src="https://github.com/user-attachments/assets/3b3bd06d-5501-4a4d-94a5-500cb2036be1" />
<img width="1800" height="900" alt="image" src="https://github.com/user-attachments/assets/0a0ab397-035c-47b9-884f-c3709c893147" />
<img width="1800" height="900" alt="image" src="https://github.com/user-attachments/assets/7a68a794-b836-44ca-af42-4180020fa554" />
<img width="1800" height="900" alt="image" src="https://github.com/user-attachments/assets/1d1d5fa9-747d-4e30-adde-c402e0975585" />
<img width="1800" height="900" alt="image" src="https://github.com/user-attachments/assets/0a993a61-a89b-4869-a2a1-f057388c6bb7" />
<img width="1800" height="900" alt="image" src="https://github.com/user-attachments/assets/6abd99a4-26ba-406a-9917-6ccf78e0b3d8" />
<img width="1800" height="900" alt="image" src="https://github.com/user-attachments/assets/445276af-b71f-4a9a-afc3-9c3c70d22f96" />
<img width="1800" height="900" alt="image" src="https://github.com/user-attachments/assets/35e4ebd3-9464-4731-bb8a-e6d1443a36b2" />
<img width="1800" height="900" alt="image" src="https://github.com/user-attachments/assets/fb36bfb3-325b-4a79-982f-4ff36cab89d1" />
<img width="1800" height="900" alt="image" src="https://github.com/user-attachments/assets/6b6f5d17-a0cb-4528-81c0-8dd2cbbab86f" />
<img width="1800" height="900" alt="image" src="https://github.com/user-attachments/assets/5712ea55-ecd0-431a-89ce-2887454aa87d" />
<img width="453" height="139" alt="image" src="https://github.com/user-attachments/assets/8a9f1923-fd98-49df-979b-68b229060994" />

# 能源
## 实验结果  
- 模型对比结果  
<img width="904" height="666" alt="image" src="https://github.com/user-attachments/assets/8bb7c2df-4233-4b09-87a0-506950ce3250" />  
- 决策树（前两层）

<img width="2211" height="1242" alt="image" src="https://github.com/user-attachments/assets/3912f857-34b9-4a9e-8f3f-79d1bd41fda8" />
