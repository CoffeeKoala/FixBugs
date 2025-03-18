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

from scipy.stats import pearsonr
from statsmodels.stats.proportion import proportions_ztest
import statsmodels.stats.weightstats as sw
from scipy import stats

matplotlib.__version__


from scipy.stats import norm

# ###############################################

import ChromaPalette as CP
from IPython.display import display_html

from matplotlib.patches import Rectangle
from matplotlib.font_manager import FontProperties

# 绘图配色list
color_list = CP.chroma_palette.color_palette(name='Sunrise',N=5)

# 作图list
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
font = FontProperties(fname='simhei.ttf')  # 使用黑体，字体文件路径需正确



# ###############################################

# # 正态分布检查
# alpha = 0.05

# rdc = rdc

# mu = df_pict_two_rdc_temp['rt'].unstack()[rdc].values.mean()
# sigma = df_pict_two_rdc_temp['rt'].unstack()[rdc].values.std()

# dist_1 = df_pict_two_rdc_temp['rt'].unstack()[rdc].values

# print("检查是否符合正太分布(双侧): \nrdc is {}\t mu is {:.4f}%\t sigma is {:.4f}%\n".format(rdc,mu*100,sigma*100))

# print("miu +- 1* sigma cnt:{}\t\t percentage{:.4f}".format( sum( dist_1 > mu + 1* sigma).astype(int), 0.5 - sum( dist_1 > mu + 1* sigma).astype(int)/len(dist_1)))
# print("miu +- 2* sigma cnt:{}\t\t percentage{:.4f}".format( sum( dist_1> mu + 2* sigma).astype(int), 0.5 -  sum( dist_1> mu + 2* sigma).astype(int)/len(dist_1)))
# print("miu +- 3* sigma cnt:{}\t\t percentage{:.4f}".format(sum( dist_1> mu + 3* sigma).astype(int) , 0.5 -  sum( dist_1> mu + 3* sigma).astype(int)/len(dist_1)))
# 非常符合正太分布！


# ###############################################
# 仓波动画图

# fig, ax = plt.subplots(figsize=(12, 10))
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 10))
ax = ax.flatten()

ax[0].plot(x, y, 'b-', lw=2, alpha=0.6, label=rdc,linewidth=2)
ax[0].plot(x, y2, 'r-', lw=2, alpha=0.6, label=rdc2)

# ax[0].plot(x, df_pict_m_rdc.loc[rdc,'rt'] * np.ones(len(x)), linestyle = '--',color = 'royalblue', lw=2, alpha=0.6, label='均值')
# ax[0].plot(x, df_pict_m_rdc.loc[rdc2,'rt'] * np.ones(len(x)), linestyle = '--',color = 'tomato', lw=2, alpha=0.6, label='均值')


ax[0].xaxis.set_major_formatter(mdate.DateFormatter('%m-%d'))
ax[0].tick_params( rotation=25)

ax[0].set_title('rt',fontsize=14)


# mu = df_pict_m_rdc['rt'].unstack()[rdc1].values.mean()
# mu2 = df_pict_m_rdc['rt'].unstack()[rdc2].values.mean()
# sigma = df_pict_m_rdc['rt'].unstack()[rdc1].values.std()
# sigma2 = df_pict_m_rdc['rt'].unstack()[rdc2].values.std()

# 横轴日期显示
# xaxis = np.linspace(norm.ppf(0.01, mu, sigma), norm.ppf(0.99, mu, sigma), 100)
# pdf = norm.pdf(xaxis, mu, sigma)

# ax[1].plot(xaxis, pdf, 'b--', lw=2, alpha=0.6, label=rdc1)
# xaxis2 = np.linspace(norm.ppf(0.01, mu2, sigma2), norm.ppf(0.99, mu2, sigma2), 100)
# pdf2 = norm.pdf(xaxis2, mu2, sigma2)

# ax[1].plot(xaxis2, pdf2, 'r--', lw=2, alpha=0.6, label=rdc2)
# ax[0].legend()
# ax[1].legend()

# plt.savefig('test.png', dpi=300)


# rt 检查单变量是否和大盘一致

# sku_list =  df_case['base_sku_name'].unique()
# rdc_list =  df_case['rdc_name'].unique()

# for rdc in rdc_list:
#     cnt = 0
#     arr = df_rdc_dt.loc[(df_rdc_dt['rdc_name'] == rdc),'rt']
#     v   = df_rdc_dt.loc[ (df_rdc_dt['rdc_name'] == rdc),'r'].sum()/ df_rdc_dt.loc[ (df_rdc_dt['rdc_name'] == rdc),'N'].sum()

#     # zscore,pvalue = sw.ztest(arr.values, value =v],alternative='two-sided')
#     zscore,pvalue = stats.ttest_1samp(arr.values,v)
#     if pvalue > 0.05:
#         result = '差错率一致稳定'
#     else:
#         result = '不一致'
#     print("当前中心仓...\t{},\t 差错率均值{:.4f},\t pvalue: {:.4f},\t T检测结果:\t{}".format(rdc,v,pvalue,result))

#     # print(df_rdc_dt.loc[ (df_rdc_dt['rdc_name'] == rdc),'r'].sum(),df_rdc_dt.loc[ (df_rdc_sku['rdc_name'] == rdc),'N'].sum())


# 皮尔森相关系数检测

# x = df_rdc_sku_dt_unstack.loc[sku_sample,'rt'].values

# if len(x) >= 2:
#     r,p = pearsonr(x,marginlines)

#     if p < 0.05:
#         if abs(r) > 0.8:
#                 result = '线性相关-极强相关'
#             elif abs(r) > 0.6 :
#                  result = '线性相关-强相关'
#             elif abs(r) > 0.4 :
#                  result = '线性相关-中相关'
#             elif abs(r) > 0.2:
#                  result = '线性相关-弱相关'
#             else:
#                 result = '线性相关-极弱相关'
#         else:
#             result = '线性不相关'


# 卡方检测
# z,p =        proportions_ztest(x,y ,alternative='two-sided')
# result = '差错率一致' if p > 0.05 else '差错率不一致'

# df_temp = pd.DataFrame.from_dict( [{'rdc': rdc
# , 'sku':sku_sample
# ,'缺陷率一致pvalue':p
# ,'缺陷率检查结果':result
# # , '单品r': r



# 销量时序稳定性检查
import statsmodels.stats.weightstats as sw
from scipy import stats
       #  if len(arr)>1 :
       #      # zscore,pvalue = sw.ztest(arr.values, value =v],alternative='two-sided')
       #      zscore,pvalue = stats.ttest_1samp(arr.values,v)

       #      if pvalue > 0.05:
       #          result = '差错率一致稳定'
       #      else:
       #          result = '不一致'

       #      df_rt_test = pd.concat([df_rt_test
       #  , pd.DataFrame.from_dict( [{'rdc':rdc
       #  , 'sku':sku_sample
       #  , '差错率t检测序列长度':len(arr)
       #  ,'差错率t检测pvalue':pvalue
       #  ,'差错率t检测':result
       # }])]


##########################################################################################
# 画图

# # 绘图
# fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))
# ax = ax.flatten()

# color_daily = ['g','r','b','orange']
# color_average = ['darkgreen','firebrick','mediumblue','darkorange']

# cnt = 0
# for sku in sells_sku:
#     ax[cnt].plot( time_x,df_pict[sku],             label='销量',color = color_daily[cnt],   linestyle = '-',marker='o',linewidth=2)
#     ax[cnt].plot( time_x,df_pict['均值-'+sku],     label='均值',color = color_average[cnt], linestyle = '--',marker='',linewidth=2)

#     # ax[cnt].set_title(sku,fontsize=14,fontfamily = 'KaiTi')
    
#     ax[cnt].set_title(sku,fontsize=14)
#     ax[cnt].set_xlabel('日期', fontsize=12)
#     ax[cnt].set_ylabel('销量', fontsize=12)
    
#     ax[cnt].tick_params(axis='x', rotation=25)
#     ax[cnt].xaxis.set_major_formatter(mdate.DateFormatter('%m-%d'))
#     ax[cnt].set_ylim([0, 4700])
#     ax[cnt].legend()
#     cnt+=1

# ax[0].text(time_x[0],3000, text[0], ha='left',fontsize=12,)
# ax[1].text(time_x[0],600, text[1], ha='left',fontsize=12,)
# ax[2].text(time_x[9],500, text[2], ha='left',fontsize=12,)
# ax[3].text(time_x[0],3000, text[3], ha='left',fontsize=12,)

# fig.suptitle('同等比例下不同商品的销量变化', fontsize=20)
# plt.savefig('销量占比检测结果.png')


#########################################################
# 画出方块

import math

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_colortable(colors, *, ncols=4, sort_colors=True):

    cell_width = 212
    cell_height = 22
    swatch_width = 48
    margin = 12

    # Sort colors by hue, saturation, value and name.
    if sort_colors is True:
        names = sorted(
            colors, key=lambda c: tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(c))))
    else:
        names = list(colors)

    n = len(names)
    nrows = math.ceil(n / ncols)

    width = cell_width * 4 + 2 * margin
    height = cell_height * nrows + 2 * margin
    dpi = 72

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(margin/width, margin/height,
                        (width-margin)/width, (height-margin)/height)
    ax.set_xlim(0, cell_width * 4)
    ax.set_ylim(cell_height * (nrows-0.5), -cell_height/2.)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()

    for i, name in enumerate(names):
        row = i % nrows
        col = i // nrows
        y = row * cell_height

        swatch_start_x = cell_width * col
        text_pos_x = cell_width * col + swatch_width + 7

        ax.text(text_pos_x, y, name, fontsize=14,
                horizontalalignment='left',
                verticalalignment='center')

        ax.add_patch(
            Rectangle(xy=(swatch_start_x, y-9), width=swatch_width,
                      height=18, facecolor=colors[name], edgecolor='0.7')
        )

    return fig

plot_colortable(mcolors.CSS4_COLORS)
plt.show()
