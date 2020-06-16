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

# ������Ⱥ������ΪN
N = 58000000
# ���ó�ʼʱ�ĸ�Ⱦ����I0Ϊ239
I0 = 239
# ���ó�ʼʱ�Ļָ�����R0Ϊ31
R0 = 31
# ���ԣ���ʼ�׸�����Ⱥ���� = ������ - ��ʼ��Ⱦ���� - ��ʼ��������
S0 = N - I0 - R0
# ���ó�ʼֵ
y0 = [S0, I0, R0]



# ���ù��������ʱ����Ϊ60��
t = np.linspace(1,60,60)

# ����betaֵ����0.125
beta = 0.125

# ����gamma��ֵ����0.05
gamma = 0.05

solution = odeint(SIR, y0, t, args = (beta, gamma))

# Ҫ��Python������������ÿ�ѧ��������ʾ
np.set_printoptions(suppress=True)

# ��������ǰ���н��в鿴
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

# ��������
data = pd.read_csv('./alltime_province_2020_05_29.csv')
# ѡ�������й��ں���ʡ������
hubei = data[data['name'] == '����']

infectious_real = hubei['total_confirm'] - hubei['total_heal'] - hubei['total_dead']
recovered_real = hubei['total_heal'] + hubei['total_dead']
susceptible_real = N - infectious_real - recovered_real

# ȷ���۲��ʱ������
T = len(infectious_real)
# ���ù��������ʱ����ΪT��
t = np.linspace(1,T,T)
# ������������������
solution = odeint(SIR, y0, t, args = (beta, gamma))
# ��ͼ
fig, ax = plt.subplots(facecolor='w', dpi=100)
# ���ƹ��Ƶ�I��������ʵ��I����
ax.plot(t, infectious_real, 'r-.', alpha=0.5, lw=2, label='infected_real')
ax.plot(t, solution[:,1], 'r', alpha=0.5, lw=2, label='infected_predict')
# ���ƹ��Ƶ�R��������ʵ��R����
ax.plot(t, recovered_real, 'g-.', alpha=0.5, lw=2, label='recovered_real')
ax.plot(t, solution[:,2], 'g', alpha=0.5, lw=2, label='recovered_predict')
# ���ú���������
ax.set_xlabel('Time/days')
ax.set_ylabel('Number')
# ���ͼ��
ax.legend()
ax.grid(axis='y')
plt.box(False)

t = np.linspace(1, 360, 360)
# ����beta��gamma����
param_list = [(0.125, 0.05),
              (0.25, 0.05),
              (0.25, 0.1),
              (0.125, 0.1)]
# Ԥ��
solution_list = [odeint(SIR, y0, t, args=item) for item in param_list]
# ��ͼ
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
    # ȷ��ѵ��ģ�͵�����
    size = len(infectious)
    # ����ʱ����
    t = np.linspace(1,size,size)
    beta, gamma = parameters
    # ����Ԥ��ֵ
    solution = odeint(SIR, y0, t, args=(beta, gamma))
    # ����ÿ�յĸ�Ⱦ��������Ԥ��ֵ����ʵֵ�ľ������
    l1 = np.mean((solution[:,1] - infectious)**2)
    # ����ÿ�յ�������������Ԥ��ֵ����ʵֵ֮��ľ������
    l2 = np.mean((solution[:,2] - recovered)**2)
    # ����SIRģ�͵���ʧֵ
    return l1+l2

data = pd.read_csv('./alltime_world_2020_04_04.csv')
# ��ѡ�����й������������������
italy = data[data['name']=='�����']

# ��ȡ1��31����3��15��֮����������������
italy_train = italy.set_index('date').loc['2020-01-31':'2020-03-15']
# ȷ��ѵ����ÿ��ĸ�Ⱦ������
infectious_train = italy_train['total_confirm'] - italy_train['total_heal'] - italy_train['total_dead']
# �뽨��SIRģ��ʱ�����ƣ���������Ҳѡȡÿ��Ŀ����ߺ���������ΪSIRģ�͵Ļָ���
recovered_train = italy_train['total_heal'] + italy_train['total_dead']

# �������˿�N = 60000000
N = 60000000
# ȷ��ѵ����ÿ����׸�������
susceptible_train = N - recovered_train - infectious_train

# ��ȡ3��16����4��3��֮����������������
italy_valid = italy.set_index('date').loc['2020-03-16':'2020-04-03']
# ȷ����֤����ÿ��ĸ�Ⱦ������
infectious_valid = italy_valid['total_confirm'] - italy_valid['total_heal'] - italy_valid['total_dead']
# ȷ����֤����ÿ�������������
recovered_valid = italy_valid['total_heal'] + italy_valid['total_dead']
# ��Ϊ���ǵ���ʧ������ֻ����I(t)��R(t),��������֤���У����ǲ��ټ����׸�������

# ģ�ͳ�ʼֵ
I0 = 2
R0 = 0
S0 = N - I0 - R0
y0 = [S0,I0,R0]

# ����minimize����
from scipy.optimize import minimize

# ѵ��ģ��
optimal = minimize(loss,[0.0001,0.0001],
                   args=(infectious_train,recovered_train,y0),
                   method='L-BFGS-B',
                   bounds=[(0.00000001, 1), (0.00000001, 1)])

beta,gamma = optimal.x
# ���beta��gammaֵ
print([beta,gamma])

# ȷ����ֵ
I0_valid = 23073
R0_valid = 4907
S0_valid = N - I0_valid- R0_valid
y0_valid = [S0_valid, I0_valid, R0_valid]



# ȷ���۲��ʱ������
T = len(infectious_valid)
# ���ù��������ʱ����ΪT��
t = np.linspace(1,T,T)
# ������������������
solution = odeint(SIR, y0_valid, t, args = (beta, gamma))
# ��ͼ
fig, ax = plt.subplots(facecolor='w', dpi=100)
# ���ƹ��Ƶ�I��������ʵ��I����
ax.plot(t, infectious_valid, 'r-.', alpha=0.5, lw=2, label='infectious_valid')
ax.plot(t, solution[:,1], 'r', alpha=0.5, lw=2, label='infectious_predict')
# ���ƹ��Ƶ�R��������ʵ��R����
ax.plot(t, recovered_valid, 'g-.', alpha=0.5, lw=2, label='recovered_valid')
ax.plot(t, solution[:,2], 'g', alpha=0.5, lw=2, label='recovered_predict')
# ���ú���������
ax.set_xlabel('Time/days')
ax.set_ylabel('Number')
# ���ͼ��
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


# ģ�ͳ�ʼֵ
def get_init_data(N, I0, R0):
    S0 = N - I0 - R0
    return [S0, I0, R0]

# ��ȡ3��8����3��15��֮����������������
italy_train = italy.set_index('date').loc['2020-03-08':'2020-03-15']
# ȷ��ѵ����ÿ��ĸ�Ⱦ������
infectious_train = italy_train['total_confirm'] - italy_train['total_heal'] - italy_train['total_dead']
# �뽨��SIRģ��ʱ�����ƣ���������Ҳѡȡÿ��Ŀ����ߺ���������ΪSIRģ�͵Ļָ���
recovered_train = italy_train['total_heal'] + italy_train['total_dead']


N = 60000000
I0 = 6534
R0 = 988
y0 = get_init_data(N, 8514, 1635)

# ����ģ�ͣ��趨beta gamma��ʼֵ���Ż�����
model = SIRModel(0.0001, 0.0001, 'L-BFGS-B')

# ѵ��ģ�ͣ������������ʼֵ��ѵ����
model.fit(y0, infectious_train, recovered_train)

# ����������Ų���
best_params = model.get_optimal_params()
print(best_params)

# 3��16�������ֵ
I0_valid = 23073
R0_valid = 4907
y0_valid = get_init_data(N, I0_valid, R0_valid)
# Ԥ��
predict_result = model.predict(y0_valid,19)

t = np.linspace(1,T,T)
# ��ͼ
fig, ax = plt.subplots(facecolor='w', dpi=100)
# ���ƹ��Ƶ�I��������ʵ��I����
ax.plot(t, infectious_valid, 'r-.', alpha=0.5, lw=2, label='infectious_valid')
ax.plot(t, predict_result[:,1], 'r', alpha=0.5, lw=2, label='infectious_predict')
# ���ƹ��Ƶ�R��������ʵ��R����
ax.plot(t, recovered_valid, 'g-.', alpha=0.5, lw=2, label='recovered_valid')
ax.plot(t, predict_result[:,2], 'g', alpha=0.5, lw=2, label='recovered_predict')
# ���ú���������
ax.set_xlabel('Time/days')
ax.set_ylabel('Number')
# ���ͼ��
ax.legend()
ax.grid(axis='y')
plt.box(False)

# ��ȡ3��31����4��3��֮����������������
italy_train = italy.set_index('date').loc['2020-03-31':'2020-04-03']
# ȷ��ѵ����ÿ��ĸ�Ⱦ������
infectious_train = italy_train['total_confirm'] - italy_train['total_heal'] - italy_train['total_dead']
# �뽨��SIRģ��ʱ�����ƣ���������Ҳѡȡÿ��Ŀ����ߺ���������ΪSIRģ�͵Ļָ���
recovered_train = italy_train['total_heal'] + italy_train['total_dead']

N = 60000000
I0 = 77635
R0 = 28157
y0 = get_init_data(N, I0, R0)

# ����ģ�ͣ��趨beta gamma��ʼֵ���Ż�����
new_model = SIRModel(0.0001, 0.0001, 'L-BFGS-B')

# ѵ��ģ�ͣ������������ʼֵ��ѵ����
new_model.fit(y0, infectious_train,recovered_train)

# ����������Ų���
best_params = new_model.get_optimal_params()

N = 60000000
I0 = 85388
R0 = 34439
y0_test = get_init_data(N, I0, R0)

# ����Ԥ��
predict_result = new_model.predict(y0_test,730)

infectious_real = italy['total_confirm'] - italy['total_heal'] - italy['total_dead']
recovered_real = italy['total_heal'] + italy['total_dead']
t = np.linspace(1,len(infectious_real),len(infectious_real))
tpredict = np.linspace(64,793,730)

fig = plt.figure(facecolor='w',dpi=100)
ax = fig.add_subplot(111)
# ������ʵ��I��������ʵ��R����
ax.plot(t, infectious_real, 'r', alpha=0.5, lw=2, label='infectious_real')
ax.plot(t, recovered_real, 'g', alpha=0.5, lw=2, label='recovered_real')
# ����Ԥ���I���ߡ�R������S����
ax.plot(tpredict, predict_result[:,1], 'r-.', alpha=0.5, lw=2, label='infectious_predict')
ax.plot(tpredict, predict_result[:,2], 'g-.', alpha=0.5, lw=2, label='recovered_predict')
ax.plot(tpredict, predict_result[:,0], 'b-.', alpha=0.5, lw=2, label='susceptible_predict')


# ���ú���������
ax.set_xlabel('Time/days')
ax.set_ylabel('Number')
# ���ͼ��
legend = ax.legend()
ax.grid(axis='y')
plt.box(False)
