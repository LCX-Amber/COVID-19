import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt

def SIR(y,t,beta,gamma):
    S,I,R = y
    dSdt = -S*(I/(S+I+R))*beta
    dIdt = beta*S*I/(S+I+R)-gamma*I
    dRdt = gamma*I
    return [dSdt,dIdt,dRdt]

# 设置人群总人数为N
N = 58000000
# 设置初始时的感染人数I0为239
I0 = 239
# 设置初始时的恢复人数R0为31
R0 = 31
# 所以，初始易感者人群人数 = 总人数 - 初始感染人数 - 初始治愈人数
S0 = N - I0 - R0
# 设置初始值
y0 = [S0, I0, R0]



# 设置估计疫情的时间跨度为60天
t = np.linspace(1,60,60)

# 设置beta值等于0.125
beta = 0.125

# 设置gamma的值等于0.05
gamma = 0.05

solution = odeint(SIR, y0, t, args = (beta, gamma))

# 要求Python的所有输出不用科学计数法表示
np.set_printoptions(suppress=True)

# 输出结果的前四行进行查看
print(solution[0:4,0:3])

fig, ax = plt.subplots(facecolor='w', dpi=100)

for data, color, label_name in zip([solution[:,1], solution[:,2]], ['r', 'g'], ['infectious', 'recovered']):
    ax.plot(t, data, color, alpha=0.5, lw=2, label=label_name)

ax.set_xlabel('Time/days')
ax.set_ylabel('Number')
ax.legend()
ax.grid(axis='y')
plt.box(False)

t = np.linspace(1,360,360)

solution = odeint(SIR, y0, t, args = (beta, gamma))

fig, ax = plt.subplots(facecolor='w', dpi=100)

for index, color, label_name in zip(range(3), ['b','r','g'], ['susceptible', 'infectious', 'recovered']):
    ax.plot(t, solution[:, index], color, alpha=0.5, lw=2, label=label_name)

ax.set_xlabel('Time/days')
ax.set_ylabel('Number')
ax.legend()
ax.grid(axis='y')
plt.box(False)

# 读入数据
data = pd.read_csv('./alltime_province_2020_05_29.csv')
# 选择数据中关于湖北省的数据
hubei = data[data['name'] == '湖北']

infectious_real = hubei['total_confirm'] - hubei['total_heal'] - hubei['total_dead']
recovered_real = hubei['total_heal'] + hubei['total_dead']
susceptible_real = N - infectious_real - recovered_real

# 确定观察的时间周期
T = len(infectious_real)
# 设置估计疫情的时间跨度为T天
t = np.linspace(1,T,T)
# 估计三种人数的数量
solution = odeint(SIR, y0, t, args = (beta, gamma))
# 绘图
fig, ax = plt.subplots(facecolor='w', dpi=100)
# 绘制估计的I曲线与真实的I曲线
ax.plot(t, infectious_real, 'r-.', alpha=0.5, lw=2, label='infected_real')
ax.plot(t, solution[:,1], 'r', alpha=0.5, lw=2, label='infected_predict')
# 绘制估计的R曲线与真实的R曲线
ax.plot(t, recovered_real, 'g-.', alpha=0.5, lw=2, label='recovered_real')
ax.plot(t, solution[:,2], 'g', alpha=0.5, lw=2, label='recovered_predict')
# 设置横纵座标轴
ax.set_xlabel('Time/days')
ax.set_ylabel('Number')
# 添加图例
ax.legend()
ax.grid(axis='y')
plt.box(False)

t = np.linspace(1, 360, 360)
# 设置beta和gamma参数
param_list = [(0.125, 0.05),
              (0.25, 0.05),
              (0.25, 0.1),
              (0.125, 0.1)]
# 预测
solution_list = [odeint(SIR, y0, t, args=item) for item in param_list]
# 绘图
fig = plt.figure(facecolor='w', figsize=(15, 10), dpi=100)

for plot_index, solution, params in zip(range(5)[1:], solution_list, param_list):

    ax = fig.add_subplot(int('22' + str(plot_index)))
    ax.set_title(r'$\beta$ = %.3f  $\gamma$ = %.3f' % params)

    for index, color, label_name in zip(range(3),
                                        ['b', 'r', 'g'],
                                        ['susceptible', 'infectious', 'recovered']):
        ax.plot(t, solution[:, index], color, alpha=0.5, lw=2, label=label_name)

    ax.set_xlabel('Time/days')
    ax.set_ylabel('Number')
    ax.legend()
    ax.grid(axis='y')
    plt.box(False)

def loss(parameters,infectious, recovered, y0):
    # 确定训练模型的天数
    size = len(infectious)
    # 设置时间跨度
    t = np.linspace(1,size,size)
    beta, gamma = parameters
    # 计算预测值
    solution = odeint(SIR, y0, t, args=(beta, gamma))
    # 计算每日的感染者人数的预测值和真实值的均方误差
    l1 = np.mean((solution[:,1] - infectious)**2)
    # 计算每日的治愈者人数的预测值和真实值之间的均方误差
    l2 = np.mean((solution[:,2] - recovered)**2)
    # 返回SIR模型的损失值
    return l1+l2

data = pd.read_csv('./alltime_world_2020_04_04.csv')
# 挑选出其中关于意大利的疫情数据
italy = data[data['name']=='意大利']

# 截取1月31日至3月15日之间的意大利疫情数据
italy_train = italy.set_index('date').loc['2020-01-31':'2020-03-15']
# 确定训练集每天的感染者人数
infectious_train = italy_train['total_confirm'] - italy_train['total_heal'] - italy_train['total_dead']
# 与建立SIR模型时相类似，这里我们也选取每天的康复者和死亡者作为SIR模型的恢复者
recovered_train = italy_train['total_heal'] + italy_train['total_dead']

# 设置总人口N = 60000000
N = 60000000
# 确定训练集每天的易感者人数
susceptible_train = N - recovered_train - infectious_train

# 截取3月16日至4月3日之间的意大利疫情数据
italy_valid = italy.set_index('date').loc['2020-03-16':'2020-04-03']
# 确定验证集的每天的感染者人数
infectious_valid = italy_valid['total_confirm'] - italy_valid['total_heal'] - italy_valid['total_dead']
# 确定验证集的每天的治愈者人数
recovered_valid = italy_valid['total_heal'] + italy_valid['total_dead']
# 因为我们的损失函数中只包含I(t)和R(t),所以在验证集中，我们不再计算易感者人数

# 模型初始值
I0 = 2
R0 = 0
S0 = N - I0 - R0
y0 = [S0,I0,R0]

# 导入minimize函数
from scipy.optimize import minimize

# 训练模型
optimal = minimize(loss,[0.0001,0.0001],
                   args=(infectious_train,recovered_train,y0),
                   method='L-BFGS-B',
                   bounds=[(0.00000001, 1), (0.00000001, 1)])

beta,gamma = optimal.x
# 输出beta、gamma值
print([beta,gamma])

# 确定初值
I0_valid = 23073
R0_valid = 4907
S0_valid = N - I0_valid- R0_valid
y0_valid = [S0_valid, I0_valid, R0_valid]



# 确定观察的时间周期
T = len(infectious_valid)
# 设置估计疫情的时间跨度为T天
t = np.linspace(1,T,T)
# 估计三种人数的数量
solution = odeint(SIR, y0_valid, t, args = (beta, gamma))
# 绘图
fig, ax = plt.subplots(facecolor='w', dpi=100)
# 绘制估计的I曲线与真实的I曲线
ax.plot(t, infectious_valid, 'r-.', alpha=0.5, lw=2, label='infectious_valid')
ax.plot(t, solution[:,1], 'r', alpha=0.5, lw=2, label='infectious_predict')
# 绘制估计的R曲线与真实的R曲线
ax.plot(t, recovered_valid, 'g-.', alpha=0.5, lw=2, label='recovered_valid')
ax.plot(t, solution[:,2], 'g', alpha=0.5, lw=2, label='recovered_predict')
# 设置横纵坐标轴
ax.set_xlabel('Time/days')
ax.set_ylabel('Number')
# 添加图例
ax.legend()
ax.grid(axis='y')
plt.box(False)


class SIRModel:
    import numpy as np
    from scipy.integrate import odeint

    def __init__(self, beta, gamma, method):
        self.__beta = beta
        self.__gamma = gamma
        self.__method = method
        self.__optimal = None
        self.__predict_loss = None

    def sir_model(self, y0, t, beta, gamma):
        S, I, R = y0
        dSdt = -beta * S * I / (S + I + R)
        dIdt = beta * S * I / (S + I + R) - gamma * I
        dRdt = gamma * I
        return [dSdt, dIdt, dRdt]

    def loss_function(self, params, infected, recovered, y0):
        size = len(infected)
        t = np.linspace(1, size, size)
        beta, gamma = params
        solution = odeint(self.sir_model, y0, t, args=(beta, gamma))
        l1 = np.mean((solution[:, 1] - infected) ** 2)
        l2 = np.mean((solution[:, 2] - recovered) ** 2)
        return l1 + l2

    def fit(self, y0, infected, recovered):
        self.__optimal = minimize(self.loss_function, [self.__beta, self.__gamma],
                                  args=(infected, recovered, y0),
                                  method=self.__method,
                                  bounds=[(0.00000001, 1), (0.00000001, 1)])

    def predict(self, test_y0, days):
        predict_result = odeint(self.sir_model, test_y0, np.linspace(1, days, days), args=tuple(self.__optimal.x))
        return predict_result

    def get_optimal_params(self):
        return self.__optimal.x

    def get_predict_loss(self):
        return self.__predict_loss


# 模型初始值
def get_init_data(N, I0, R0):
    S0 = N - I0 - R0
    return [S0, I0, R0]

# 截取3月8日至3月15日之间的意大利疫情数据
italy_train = italy.set_index('date').loc['2020-03-08':'2020-03-15']
# 确定训练集每天的感染者人数
infectious_train = italy_train['total_confirm'] - italy_train['total_heal'] - italy_train['total_dead']
# 与建立SIR模型时相类似，这里我们也选取每天的康复者和死亡者作为SIR模型的恢复者
recovered_train = italy_train['total_heal'] + italy_train['total_dead']


N = 60000000
I0 = 6534
R0 = 988
y0 = get_init_data(N, 8514, 1635)

# 建立模型，设定beta gamma初始值，优化方法
model = SIRModel(0.0001, 0.0001, 'L-BFGS-B')

# 训练模型，输入参数：初始值，训练集
model.fit(y0, infectious_train, recovered_train)

# 输出估计最优参数
best_params = model.get_optimal_params()
print(best_params)

# 3月16日疫情初值
I0_valid = 23073
R0_valid = 4907
y0_valid = get_init_data(N, I0_valid, R0_valid)
# 预测
predict_result = model.predict(y0_valid,19)

t = np.linspace(1,T,T)
# 绘图
fig, ax = plt.subplots(facecolor='w', dpi=100)
# 绘制估计的I曲线与真实的I曲线
ax.plot(t, infectious_valid, 'r-.', alpha=0.5, lw=2, label='infectious_valid')
ax.plot(t, predict_result[:,1], 'r', alpha=0.5, lw=2, label='infectious_predict')
# 绘制估计的R曲线与真实的R曲线
ax.plot(t, recovered_valid, 'g-.', alpha=0.5, lw=2, label='recovered_valid')
ax.plot(t, predict_result[:,2], 'g', alpha=0.5, lw=2, label='recovered_predict')
# 设置横纵坐标轴
ax.set_xlabel('Time/days')
ax.set_ylabel('Number')
# 添加图例
ax.legend()
ax.grid(axis='y')
plt.box(False)

# 截取3月31日至4月3日之间的意大利疫情数据
italy_train = italy.set_index('date').loc['2020-03-31':'2020-04-03']
# 确定训练集每天的感染者人数
infectious_train = italy_train['total_confirm'] - italy_train['total_heal'] - italy_train['total_dead']
# 与建立SIR模型时相类似，这里我们也选取每天的康复者和死亡者作为SIR模型的恢复者
recovered_train = italy_train['total_heal'] + italy_train['total_dead']

N = 60000000
I0 = 77635
R0 = 28157
y0 = get_init_data(N, I0, R0)

# 建立模型，设定beta gamma初始值，优化方法
new_model = SIRModel(0.0001, 0.0001, 'L-BFGS-B')

# 训练模型，输入参数：初始值，训练集
new_model.fit(y0, infectious_train,recovered_train)

# 输出估计最优参数
best_params = new_model.get_optimal_params()

N = 60000000
I0 = 85388
R0 = 34439
y0_test = get_init_data(N, I0, R0)

# 进行预测
predict_result = new_model.predict(y0_test,730)

infectious_real = italy['total_confirm'] - italy['total_heal'] - italy['total_dead']
recovered_real = italy['total_heal'] + italy['total_dead']
t = np.linspace(1,len(infectious_real),len(infectious_real))
tpredict = np.linspace(64,793,730)

fig = plt.figure(facecolor='w',dpi=100)
ax = fig.add_subplot(111)
# 绘制真实的I曲线与真实的R曲线
ax.plot(t, infectious_real, 'r', alpha=0.5, lw=2, label='infectious_real')
ax.plot(t, recovered_real, 'g', alpha=0.5, lw=2, label='recovered_real')
# 绘制预测的I曲线、R曲线与S曲线
ax.plot(tpredict, predict_result[:,1], 'r-.', alpha=0.5, lw=2, label='infectious_predict')
ax.plot(tpredict, predict_result[:,2], 'g-.', alpha=0.5, lw=2, label='recovered_predict')
ax.plot(tpredict, predict_result[:,0], 'b-.', alpha=0.5, lw=2, label='susceptible_predict')


# 设置横纵坐标轴
ax.set_xlabel('Time/days')
ax.set_ylabel('Number')
# 添加图例
legend = ax.legend()
ax.grid(axis='y')
plt.box(False)
