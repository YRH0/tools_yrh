import json
import matplotlib.pyplot as plt



'''
# 输入词典 输出柱状图    
'''
def make_graph(my_dict: dict):
    with open(my_dict,"r",encoding="utf-8") as f:
        my_dict = json.loads(f.readline())
    # 绘制柱状图
    keys = my_dict.keys()
    values = my_dict.values()
    plt.bar(keys, values)
    # 设置横轴标签和纵轴标签
    # plt.text(x=100, y=100, s='text', rotation=90)
    plt.xlabel('Fruit')
    plt.ylabel('Quantity')
    # plt.xlim(0,1000)
    plt.ylim(0,20000)
    # 设置标题
    plt.title('Quantity of Fruits')
    # 展示图形
    plt.savefig('savefig_example.png')