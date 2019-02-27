# 对错误问题的一些思考与记录


## 错误描述

将ovk改成三维后 没有好的学习结果，与用div的kernel去学习curl-free场的结果类似

但是经过检验，确实是div-free的场，但是表现的result接近0，ridge正常的回归却表现很好。

## 可能错误原因

1. 数据集错误
  数据集可能inputs和targets不对应？或者与程序要求不符合
  数据量取的太稀疏了还是太密集了？
  也有可能是某个过程需要网格化然后错误了
  
  
  
2.程序有些地方不对，因为并没有看到过用这个包做三维kernel的例子


## 初步定的解决方法

综上 都需要 将ridge 和 kernel方法差异的地方比较下才能理解