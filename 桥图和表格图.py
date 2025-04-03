# from pytalos.client import AsyncTalosClient
import numpy as np
import pandas as pd
import os,time,sys
import seaborn as sns
import scipy.stats.contingency as cy 
from sklearn.cluster import KMeans
import math
import operator
import scipy.stats
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.dates as mdate

from datetime import datetime
import networkx as nx
import Levenshtein

import ChromaPalette as CP
from IPython.display import display_html

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

color_list = CP.chroma_palette.color_palette(name='Sunrise',N=5)



# 颜色挑选模块,
colorlist = [   'Coral'
    , 'VanGogh'
    , 'CafeTarrence'
    , 'MintGradient'
    , 'Candies'
    , 'Macarons'
    , 'Matcha'
    , 'Melody'
    , 'Lollipop'
    , 'Sunrise'
    , 'Monet'
    , 'Waterlilies'
    , 'Sunflowers'
    , 'Irises'
    , 'WheatFields'
    , 'Serene'
    , 'Elegant'
    , 'Vibrance'
    , 'Radiant'
    , 'RoseGradient'
    , 'PurpleGradient'
    , 'BlueGradient'
    , 'Retro'
    , 'HarmonyMix'
    , 'SimplePastel'
    , 'SoftVintage'
    , 'EasternHues'
    , 'RelaxingPastel'
    , 'Lotus'
    , 'CalmBalance'
    , 'Eggplant'
    , 'SeaSalt'
    , 'Enamel'
    , 'Pearwood'
    , 'VintageBlend'
    , 'CozyBlue'
    , 'OrangeLatte'
    , 'WatermelonSoda'
    , 'SoftSerenity'
    , 'Westminster'
    , 'EarthyTones'
    , 'GracefulHues'
    , 'Pond'
    , 'MutedBlend'
    , 'FlowerBed'
    , 'Turner'
    , 'WarmHues'
    , 'ColorBlocking'
    , 'Porcelain'
    , 'RetroComfort'
]
s = 'EasternHues'
k = 10

fig, ax = plt.subplots(figsize = (10,6))
clist = CP.chroma_palette.color_palette(name=s,N=k)
N = 10
x = np.linspace(-10,10, N)
y = np.ones(len(x))

for i in range(k):
    noise = np.random.randn(N)/N
    ax.plot(x,i+y +noise, color = clist[i], linewidth = 2,alpha=1
            , label =  str(i)+ ' : ' + clist[i]
            , linestyle = '--', marker = 'o', markersize = 10)
ax.legend()
   
ax.set_title('配色  '+s ,fontfamily = 'KaiTi',fontsize=20)
ax.set_xlabel('n-point', fontsize=16,fontfamily = 'KaiTi')
ax.set_ylabel('RDC缺陷率', fontsize=16,fontfamily = 'KaiTi')
plt.grid(color='#323E6E', linewidth=0.5,linestyle = ':')
print(s,clist)



# 时间刻度

from matplotlib.ticker import FuncFormatter
# 自定义刻度


def custom_ticks(x, pos):
    #     到时刻
    t= x *5- 30
    if t <0:
        t = t+24*60
        h = int(t//60)
        minute = int(t - int(t/60)*60)
    else:
        h = int(t//60)
        minute = int(t - int(t/60)*60)       
    return "{:02}".format(h) + ':' + "{:02}".format(minute)


# 画出不用颜色的色块
# # 创建画布和子图



rectanglelist = []
workshop_min = 1000
workshop_max  = 0
time_min = 1000
time_max = 0

for index,row in df_cond.iterrows():
    print(row['资源层-中心仓'],row['month'])
    fig, ax = plt.subplots(figsize = (25,20))

    df_sample = df_nanning.loc[(df_nanning['资源层-中心仓'] == row['资源层-中心仓'])
                               &(df_nanning['month'] == row['month']),:].copy()
    
    
    total_workshop_cnt = df_sample['工作台编号'].nunique()
    fantaicnt = 0
    rectanglelist = []
    workshop_min = 1000
    workshop_max  = 0
    time_min = 1000
    time_max = 0
    grid_list = df_sample['网格站'].unique()

    for s in grid_list:
        workshop = df_sample.loc[df_sample['网格站'] == s,'工作台编号'].unique()
        fantaicnt = fantaicnt + len(workshop)
        c = color_dict[s]

        workshop_min = min(workshop_min,min(workshop))
        workshop_max = max(workshop_max,max(workshop))
        n_workshop = len(workshop)
        for x in workshop:
        # # 创建一个长方形，指定左下角的点、宽度和高度 # Rectangle((x, y), width, height)
            location_x = df_sample.loc[(df_sample['网格站'] == s) & (df_sample['工作台编号'] == x),'时间序列编号min'].min()
            location_y = df_sample.loc[(df_nanning['网格站'] == s) & (df_sample['工作台编号'] == x),'工作台编号'].min()
            width =  df_sample.loc[(df_sample['网格站'] == s) & (df_sample['工作台编号'] == x),'len'].min()
            end_x = df_sample.loc[(df_sample['网格站'] == s) & (df_sample['工作台编号'] == x),'时间序列编号max'].max()+1

            time_min = time_min if time_min <location_x   else location_x 
            time_max = time_max if time_max >location_x + width else location_x + width
            rectangle = Rectangle((location_x,location_y), width -0.1 ,0.8
                                  , facecolor = c
                                  , edgecolor='grey', linewidth=0.6,alpha=0.6)
            # # 将长方形添加到子图中
            rectanglelist.append(rectangle)

            text_x = location_x +width/2
            test_y = location_y

            content = '{s}, 台数{n}, 开始{st},结束{enddt},单台时长{dur}'.format(
            s = s[0:5], n = n_workshop, st = location_x*5-30 ,enddt = end_x*5-30,dur = width*5)

            ax.text(text_x, test_y + 0.1, content, fontsize=10,
                    horizontalalignment='center',verticalalignment='bottom',fontfamily = 'KaiTi')

    fantai_rt = '{:.0%}'.format(round(fantaicnt/total_workshop_cnt,3))

    for a in rectanglelist:
        ax.add_patch(a)

    print("-"*60)
    # print('时间轴范围即x轴',time_min,time_max)
    # print('工作台范围即y轴',workshop_min,workshop_max)

    ax.set_xlim(time_min-2, time_max)
    ax.set_ylim(workshop_min-1, workshop_max+1)

    ax.set_title('翻台时网格对工作台占用序列,当前中心仓及温层:  '+  '南宁一三仓60站-冷冻, 翻台率:' +fantai_rt ,fontfamily = 'KaiTi',fontsize=18)
    ax.set_xlabel('时间轴', fontsize=18,fontfamily = 'KaiTi')
    ax.set_ylabel('工作台序号', fontsize=18,fontfamily = 'KaiTi')
    plt.grid(color='#323E6E', linewidth=0.5,linestyle = ':')

    x_min, x_max = ax.get_xlim()

    print("x轴坐标范围:", x_min, "到", x_max)
    ax.xaxis.set_major_formatter(FuncFormatter(custom_ticks))
    plt.savefig('./pict/' + row['资源层-中心仓'] + str(row['month']) +'.png',dpi = 400)
    print("-"*60)



# 使用FuncFormatter应用自定义刻度

# 桑基图

import plotly.graph_objects as go

# 定义节点和流量数据
nodes = ["A", "B", "C", "D", "E"]  # 所有节点标签
source = [0, 0, 1, 2, 3]           # 源节点索引（对应nodes列表）
target = [2, 3, 4, 4, 4]           # 目标节点索引
value =  [8, 4, 6, 2, 10]          # 流量值

# 创建桑基图
fig = go.Figure(go.Sankey(
    node=dict(
        pad=10 ,          # 节点间距
        thickness=10,    # 节点宽度
        label=nodes,     # 节点标签
        color="skyblue"  # 节点颜色
    ),
    link=dict(
        source=source,   # 源节点
        target=target,   # 目标节点
        value=value,     # 流量值
        color="rgba(255,0,0,0.25)"  # 连接线颜色（半透明红色）
    )
))

# 调整布局并显示
fig.update_layout(title="桥图示例 - 数据流动桑基图", font_size=10)
fig.show()



import plotly.graph_objects as go

# 示例数据（假设为某公司月度利润构成）
categories = ["收入", "成本", "运营费用", "税费", "其他调整", "净利润"]
values = [1200, -600, -300, 900, 150, None]  # None表示自动计算累计值

measure = ["relative", "relative", "relative", "total", "relative", "total"]

# 创建瀑布图
fig = go.Figure(go.Waterfall(
    name="2023年利润",
    orientation="v", ## 'h'
    x= categories,
    y= values,
#     textposition="auto",
     textposition= 'outside',
    
    text=[f"+{v}" if v > 0 else str(v) for v in values[:-1]] + ["Total"],
    
    connector={"mode":"between","line": {"color": "gray" ,"dash":"solid"}},
    base = 0,
    measure=measure,
    opacity = 0.75, ## 整体配色透明度
    increasing={"marker": {"color": '#80B3B4', "line":{"color":"#327368", "width":2}}},   # 正值为绿色
    decreasing={"marker": {"color": '#AF6F45', "line":{"color":'#BE4C49', "width":2}}},     # 负值为红色
    totals={"marker": {"color": '#50543F','line':{'color':'#50543F','width':2}}}         # 总值为蓝色
))

# 调整布局
fig.update_layout( 
title={'text' : "利润构成瀑布图（单位：万元）"
, 'font':{'color':'blue', 'family': 'KaiTi','size':18}
,'x': 0.5 
,'y': 0.85
,'xanchor': 'center'
}
      
# , xaxis_title="item"
    
, xaxis={
        'title':  {'text': '项目','font':{'family':'KaiTi'}},
        'tickvals': [0, 1, 2, 3, 4, 5],  # 显式指定刻度位置
        'ticktext': categories,          # 显式指定刻度标签
        'tickangle': 0,                # 标签旋转角度
        'showgrid': False                # 隐藏网格线
, 'tickfont': {
    'family': "KaiTi",
    'size': 12  ,
    'color': 'darkgreen'
    }   

}
    
, yaxis_title="amt"
, showlegend=False
, waterfallgap = 0.35 ## 柱子和空格占比

)

fig.add_annotation(
    x=0.7,
    y=0.5,
    text="注：税费包含企业所得税和增值税",
    showarrow=False,
    xref="paper",
    yref="paper",
    font={'size': 12, 'color': 'black','family':'KaiTi'}
)


# 显示图表
fig.show()

