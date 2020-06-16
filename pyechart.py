# 读取数据
import pandas as pd

# # 调整配置项
import pyecharts.options as opts
# # Map类用于绘制地图
from pyecharts.charts.basic_charts import map
from pyecharts.charts.basic_charts.pie import Pie
from datetime import datetime,timedelta
import matplotlib as plt
import matplotlib.animation as animation
from IPython.display import HTML
import matplotlib.ticker as ticker

# 创建中文列名字典
name_dict = {'date': '日期', 'name': '名称', 'id': '编号', 'lastUpdateTime': '更新时间',
             'today_confirm': '当日新增确诊', 'today_suspect': '当日新增疑似',
             'today_heal': '当日新增治愈', 'today_dead': '当日新增死亡',
             'today_severe': '当日新增重症', 'today_storeConfirm': '当日现存确诊',
             'total_confirm': '累计确诊', 'total_suspect': '累计疑似','total_input':'累计输入病例',
             'total_heal': '累计治愈', 'total_dead': '累计死亡', 'total_severe': '累计重症',
             'today_input': '当日输入病例', 'total_severe': '累计重症'}


# 绘制世界各国现存确诊人数地图
world_data = pd.read_csv('./today_world_2020_05_11.csv')
world_data = world_data.rename(columns=name_dict)
world_data['当日现存确诊'] = world_data['累计确诊']-world_data['累计治愈']-world_data['累计死亡']
# contry_name = pd.read_csv('./input/county_china_english.csv', encoding='GB2312')

heatmap_data = world_data[['名称','当日现存确诊']].values.tolist()
print(heatmap_data[:10])

map_ = map().add(series_name = "现存确诊人数", # 设置提示框标签
                 data_pair = heatmap_data, # 输入数据
                 maptype = "world", # 设置地图类型为世界地图
                 is_map_symbol_show = False)# 不显示标记点

# 设置系列配置项
map_.set_series_opts(label_opts=opts.LabelOpts(is_show=False))  # 不显示国家（标签）名称

# 设置全局配置项
map_.set_global_opts(title_opts = opts.TitleOpts(title="世界各国家现存确诊人数地图"), # 设置图标题
                     # 设置视觉映射配置项
                     visualmap_opts = opts.VisualMapOpts(pieces=[ # 自定义分组的分点和颜色
                                                               {"min": 10000,"color":"#800000"}, # 栗色
                                                               {"min": 5000, "max": 9999, "color":"#B22222"}, # 耐火砖
                                                               {"min": 999, "max": 4999,"color":"#CD5C5C"}, # 印度红
                                                               {"min": 100, "max": 999, "color":"#BC8F8F"}, # 玫瑰棕色
                                                               {"max": 99, "color":"#FFE4E1"}, # 薄雾玫瑰
                                                              ],
                     is_piecewise = True))  # 显示分段式图例

map_.show()



# 绘制世界各国累计死亡人数玫瑰图
need_data = world_data[['名称', '累计死亡']][world_data['累计死亡'] >500]
rank = need_data[['名称', '累计死亡']].sort_values(by='累计死亡', ascending=False).values

pie = Pie().add(series_name = "累计死亡人数", # 添加提示框标签
                data_pair = rank, # 输入数据
                radius = ["20%", "80%"],  # 设置内半径和外半径
                center = ["60%", "60%"],  # 设置圆心位置
                rosetype = "radius")   # 玫瑰图模式，通过半径区分数值大小，角度大小表示占比

pie.set_global_opts(title_opts = opts.TitleOpts(title="世界国家累计死亡人数玫瑰图",  # 设置图标题
                                                pos_right = '40%'),  # 图标题的位置
                    legend_opts = opts.LegendOpts( # 设置图例
                                                orient='vertical', # 垂直放置图例
                                                pos_right="85%", # 设置图例位置
                                                pos_top="15%"))

pie.set_series_opts(label_opts = opts.LabelOpts(formatter="{b} : {d}%")) # 设置标签文字形式为（国家：占比（%））

# 在notebook中进行渲染
pie.render_notebook()



# 绘制三月份各国累计确证人数动态条形图
alltime_data = pd.read_csv('./alltime_world_2020_05_29.csv')
country_list = ['美国', '意大利', '中国', '西班牙', '德国', '伊朗', '法国', '英国', '瑞士','比利时']
need_data = alltime_data[alltime_data['name'].isin(country_list)]

time_list = [(datetime(2020, 3, 1) + timedelta(i)).strftime('%Y-%m-%d') for i in range(31)]

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['figure.dpi'] = 100
color_list = ['brown','peru','orange','blue','green',
              'red','yellow','teal','pink','orchid']
country_color = pd.DataFrame()
country_color['country'] = country_list
country_color['color'] = color_list


def barh_draw(day):
    # 提取每一天的数据
    draw_data = need_data[need_data['date'] == day][['name', 'total_confirm']].sort_values(by='total_confirm',
                                                                                           ascending=True)

    # 清空当前的绘图
    ax.clear()

    # 绘制条形图
    ax.barh(draw_data['name'], draw_data['total_confirm'],
            color=[country_color[country_color['country'] == i]['color'].values[0] for i in draw_data['name']])

    # 数值标签的间距
    dx = draw_data['total_confirm'].max() / 200

    # 添加数值标签
    for j, (name, value) in enumerate(zip(draw_data['name'], draw_data['total_confirm'])):
        ax.text(value + dx, j, f'{value:,.0f}', size=10, ha='left', va='center')

    # 添加日期标签
    ax.text(draw_data['total_confirm'].max() * 0.75, 0.4, day, color='#777777', size=40, ha='left')

    # 设置刻度标签的格式
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

    # 设置刻度的位置
    ax.xaxis.set_ticks_position('top')

    # 设置刻度标签的颜色和大小
    ax.tick_params(axis='x', colors='#777777', labelsize=15)

    # 添加网格线
    ax.grid(which='major', axis='x', linestyle='-')

    # 添加图标题
    ax.text(0, 11, '3月世界各国家累计确诊人数动态条形图', size=20, ha='left')

    # 去除图边框
    plt.box(False)

    # 关闭绘图框
    plt.close()

fig, ax = plt.subplots(figsize=(12, 8))



animator = animation.FuncAnimation(fig, barh_draw, frames=time_list, interval=200)
HTML(animator.to_jshtml())