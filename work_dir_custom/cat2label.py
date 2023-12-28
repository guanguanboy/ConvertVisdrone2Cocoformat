CLASSES = ('Car', 'Pedestrian', 'Cyclist')
    # 类别反查表
cat2label = {k: i for i, k in enumerate(CLASSES)}
print(cat2label)