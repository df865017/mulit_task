# mulit_task

## 简介
本项目是对多目标点击和付费任务的优化，试用了MMoE和ESMM方法，同时优化点击和付费两个目标。
其中ESMM训练AUC更好。

## 语言
Python 3.7 + Tensorflow 2.0


## 执行结果
- MMoE
  <p align="center"> <img src="https://github.com/df865017/mulit_task/blob/main/pic/mmoe.png" width="70%"> </p>
- ESMM
  <p align="center"> <img src="https://github.com/df865017/mulit_task/blob/main/pic/essm.png" width="70%"> </p>

## 代码介绍
utils是工具层，构建数据处理和模型训练和预测接口
models是模型层，构建模型的搭建
local_run_test是主程序
