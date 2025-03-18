backup for some colors

# import packages

from datetime import datetime, timedelta
from IPython.display import display_html
from pylab import rcParams
from pytalos.client import AsyncTalosClient
from scipy.optimize import curve_fit
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_boston
from sklearn.datasets import make_blobs
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
import ChromaPalette as CP
import Levenshtein
import math
import matplotlib.dates as mdate
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import operator
import os,time,sys
import pandas as pd
import scipy.stats
import scipy.stats.contingency as cy
import seaborn as sns



X_train = df[xlist]
y = df[' y ']

# 创建模型并训练
lr = LinearRegression()
lr.fit(X_train,y)
y_pred = (lr.predict(X_train)).round(0)
print('='*80)

# print("系数:", lr.coef_[0])
# df_result = pd.DataFrame(list(zip(xlist,lr.coef_[0])),columns = ['Var','Coef'])
# display_html(df_result)

print("截距项:", lr.intercept_)
print(xlist)
print('各特征权重：\t', lr.coef_)
print("截距项:", lr.intercept_)
# 计算R2值
r2 = r2_score(y, y_pred)
print('R2:\t{:.4f}'.format(r2))
print('建模master维度',X_train.shape)
print('='*80)

df_m['y_predict'] = y_pred
df_m['y_bin'] =  pd.qcut(df_m['ODR100'],20,labels =  [ i for i in range(0,20)])
df_m['y_pred_fenzi'] = ((df_m['N'] *  df_m['y_predict'])/10000).astype(int)
df_check = df_m.groupby(['y_bin'])[['N','r','y_pred_fenzi']].sum().reset_index()
df_check ['pct'] = ( df_check ['N']/df_check ['N'].sum()*100).round(2)
df_check['y'] = (df_check['r']/ df_check['N']).round(6)*10000
df_check['y_pred'] = (df_check['y_pred_fenzi']/ df_check['N']).round(6)*10000
df_check = df_check.sort_values(by = 'y',ascending=False)
# df_check
plt.rcParams['font.sans-serif'] = ['KaiTi'] # 使用中文字体\
fig, ax = plt.subplots(figsize = (12,6))
color_list = CP.chroma_palette.color_palette(name='Sunrise',N=5)
# ['#8386A8', '#C16477', '#E6A14B', '#CCC63B', '#8AB95F', '#78B9D2']
y_pict_real = df_check['y'].values
y_pict_pred = df_check['y_pred'].values
pct = df_check['pct'].values
x = list(range(len(y_pict_real)))
plt.plot(x,y_pict_real, linestyle = '-', linewidth=2,marker = 'o', markersize=6,alpha = 0.8, color='#B4666D',label = '品质ODR-真实')
plt.plot(x,y_pict_pred, linestyle = '--', linewidth=2,marker = 'o', markersize=6, color='#78B9D2',label = '品质ODR-预测')
# ax.set_ylim([0,25])
ax.set_ylabel('品质ODR * 10000',fontsize=16)
ax.set_xlabel('模型分组',fontsize= 16)
ax2 = ax.twinx()
ax2.bar(x,pct,color='#7AB7B5',alpha = 0.4,label = '样本占比',width = 0.7)
ax2.set_ylim([0,((pct.max()/10).astype(int)+2)*10])
ax2.set_ylabel('样本占比',fontsize=16,color='#7AB7B5')
plt.grid(color='#323E6E', linewidth=0.5,linestyle = ':') ## 格子的线性和颜色
ax.legend(loc='upper right',fontsize=16)
ax2.legend(loc='upper center',fontsize=16)
plt.title('多元线性回归模型性能(分组数=20)',fontsize=16)
anno_text = 'R2 = {:.2f}'.format(r2)
plt.text(np.mean(x[-6:-2]),y.mean().round(2), anno_text, ha='left',fontsize=12,fontfamily = 'consolas')
# plt.text(x[-6:-2].mean().round(2),y.min().round(2), anno_text, ha='left',fontsize=12,fontfamily = 'consolas')
plt.savefig('./PICT/模型性能.png',dpi = 250)



############################################################

N = 20
var = 'var'
df_m[var+'bin'] =  pd.qcut(df_m['var'],N,labels =  [ i for i in range(0,20)]).astype(int)
df_check = df_m.groupby([var+'bin'])[['N','r','x']].sum().reset_index()
df_check ['pct'] = ( df_check ['N']/df_check ['N'].sum()*100).round(2)
df_check ['ODR'] = ( df_check ['r']/df_check ['N']*10000).round(2)
df_check ['x'] = ( df_check ['N']/df_check ['x']).astype(int)
df_check.sort_values(by = 'bin')

x = df_check['x'].values
y = df_check['ODR'].values
pct = df_check['pct'].values

para,pcov=curve_fit(Fun,x,y)
y_fitted = Fun(x,para[0],para[1]) # 画出拟合后的曲线
fig, ax = plt.subplots(figsize = (12,6))

ax.plot(x,y, linestyle = '-', linewidth=2,marker = 'o', markersize=6,alpha = 0.9, color='#B4666D',label = '实际')
ax.plot(x,y_fitted, linestyle = '--', linewidth=2,marker = 'o', markersize=4, color='#B4666D',alpha = 0.4,label = '拟合')

ax.grid(color='#323E6E', linewidth=0.5,linestyle = ':') ## 格子的线性和颜色
ax.legend( loc='upper left',fontsize=14)

for i, v in enumerate(y): ## 标注折线数据
    ax.annotate(str(round(v,1))+'', xy=(x[i], v+0.5),color ='#B4666D',fontsize=10)

ax.set_xlabel(var,fontsize= 16)
ax.set_ylabel('品质ODR * 10000',fontsize= 14)
r2 = r2_score(y, y_fitted)

anno_text = 'y = {:.2f}*x + {:.2f}\n\nR2 = {:.2f}'.format(para[0],para[1],r2)
ax.text(x[-6:-2].mean().round(2),15, anno_text, ha='left',fontsize=12,fontfamily = 'consolas')

ax2 = ax.twinx()
w = (x.max() - x.min())/len(x)
ax2.bar(x,pct,color='#7AB7B5',alpha = 0.4,label = '样本占比',width =w/2)
ax2.set_ylim([0,((pct.max()/10).astype(int)+2)*10])
ax2.set_ylabel('样本占比',fontsize=16,color='#7AB7B5')
ax2.set_xticklabels(x)


# 增加注释
for i, v in enumerate(pct):
    ax2.annotate(str(round(v))+'%', xy=(x[i]-0.01,v- 1),color ='#486F65',fontsize=10)


# # # ax2.tick_params(axis='y', labelcolor='red')

plt.grid(color='#323E6E', linewidth=0.5,linestyle = ':') ## 格子的线性和颜色

plt.title('R{}关系'.format(var),fontsize=16)
plt.savefig('./PICT/R和{}关系.png'.format(var),dpi = 250)



##################################
# 散点图

fig, ax = plt.subplots(figsize = (20,16))

ax.set_xlim(0.2,0.9)
ax.set_ylim(0,200)

ax.scatter(x1,y1, s = 10, alpha=1, color = 'red',marker='*',label='a')
ax.scatter(x2,y2, s = 10, alpha=0.8, color = 'orange' ,marker='.',label='b')

ax.scatter(x3,y3, s = 10, alpha=0.6, color = 'blue' ,marker='.',label='c')
ax.scatter(x4,y4, s = 10, alpha=0.6, color = 'green', marker='.',label='d')

plt.grid(color='#323E6E', linewidth=0.5,linestyle = ':')
plt.legend()

##################################
# 颜色挑选模块,
# 颜色挑选模块,
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
s =  'Lollipop'
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



######################## ######################## ######################## ######################## ######################## ########################
######################## ######################## ######################## ######################## ######################## ########################
######################## ######################## ######################## ######################## ######################## ########################
######################## ######################## ######################## ######################## ######################## ########################


# 格子作图

# # 创建画布和子图

rectanglelist = []

workshop_min = 1000
workshop_max  = 0
time_min = 1000
time_max = 0

for index,row in df_cond.iterrows():
    print(row['r'],row['month'])
    fig, ax = plt.subplots(figsize = (25,20))

    df_sample = df_nanning.loc[(df_nanning['r'] == row['r'])
                               &(df_nanning['month'] == row['month']),:].copy()


    total_workshop_cnt = df_sample['no'].nunique()
    fantaicnt = 0
    rectanglelist = []
    workshop_min = 1000
    workshop_max  = 0
    time_min = 1000
    time_max = 0
    grid_list = df_sample['g'].unique()

    for s in grid_list:
        workshop = df_sample.loc[df_sample['g'] == s,'no'].unique()
        fantaicnt = fantaicnt + len(workshop)
        c = color_dict[s]

        workshop_min = min(workshop_min,min(workshop))
        workshop_max = max(workshop_max,max(workshop))
        n_workshop = len(workshop)
        for x in workshop:
        # # 创建一个长方形，指定左下角的点、宽度和高度 # Rectangle((x, y), width, height)
            location_x = df_sample.loc[(df_sample['g'] == s) & (df_sample['No'] == x),'时间序列编号min'].min()
            location_y = df_sample.loc[(df_nanning['g'] == s) & (df_sample['No'] == x),'No'].min()
            width =  df_sample.loc[(df_sample['g'] == s) & (df_sample['No'] == x),'len'].min()
            end_x = df_sample.loc[(df_sample['g'] == s) & (df_sample['No'] == x),'时间序列编号max'].max()+1

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

    ax.set_title('工作序列:  '+  'r:' +fantai_rt ,fontfamily = 'KaiTi',fontsize=18)
    ax.set_xlabel('时间轴', fontsize=18,fontfamily = 'KaiTi')
    ax.set_ylabel('工作台序号', fontsize=18,fontfamily = 'KaiTi')
    plt.grid(color='#323E6E', linewidth=0.5,linestyle = ':')

    x_min, x_max = ax.get_xlim()

    print("x轴坐标范围:", x_min, "到", x_max)
    ax.xaxis.set_major_formatter(FuncFormatter(custom_ticks))
    plt.savefig('./pict/' + row['rdc'] + str(row['month']) +'.png',dpi = 400)
    print("-"*60)



# 画图挑选配色代码
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
