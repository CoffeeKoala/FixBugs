DF.from_dict(dict) 在遇到字典键值为list时：

1. 若list 为空，则转化失败；
2. 若list为多个值，则自动拆分为多条；
