import numpy as np
import matplotlib.pyplot as plt

a = 2  # mcm
""" Аналоговая функция фильтра"""
def H(X, coef=1):
    y = []
    for x in X:
        if np.abs(x) <= 50 / coef:
            y.append(1.0)
        elif np.abs(x) > 100 / coef:
            y.append(0.0)
        else:
            y.append(0.5)
    return y

grid = np.arange(0, 120)
H1 = H(grid)
plt.figure()
plt.plot(grid, H1, linewidth=3, color='#008cf0')
plt.suptitle('Аналоговая функция цифрового фильтра')
plt.xlabel('nu, 1/mm')
plt.ylabel('H(nu)')
plt.ylim([0, 1.2])
plt.xlim([0, 120])
plt.grid()
plt.show()

""" Частота Найквиста"""
nu_N = 1 / (2 * 0.002)
print('Частота Найквиста = %0.2f 1/mm' % nu_N)

""" Нормированная аналоговая функция фильтра"""
grid_norm = np.arange(0, 1, 1 / nu_N)
H1 = H(grid_norm, nu_N)

""" Дискретные значения"""
grid_d = np.arange(0,0.7,0.3)
H_d = H(grid_d, nu_N)
print('Дискретные отсчеты: \n', grid_d, '\n Матрица разложения: \n', H_d)

plt.figure()
plt.bar(grid_d, H_d, width=0.01, color='#008cf0')
plt.scatter(grid_d, H_d, linewidth=5, color='#008cf0')
plt.plot(grid_norm, H1, linewidth=3, color='#008cf0')

plt.suptitle('Аналоговая нормированная функция цифрового фильтра')
plt.xlim([0, 1])
plt.ylim([0, 1.2])
plt.xlabel('nu')
plt.ylabel('H(nu)')
plt.grid()
plt.show()

m = np.ones([3, 3])
for i in range(3):
    for j in range(1, 3):
        m[i, j] = 2 * np.cos(j * np.pi * grid_d[i])
print(m)

""" Коэффициенты фильтра"""
h = np.linalg.solve(m, H_d)
checked_H = np.dot(m, np.reshape(h, [-1, 1]))

""" Формирование одномерного массива"""
h1d = np.concatenate((np.flip(h[1:], 0), h))
grid_d2 = np.concatenate((-np.flip(grid_d[1:], 0), grid_d))
plt.figure()
plt.bar(np.arange(5), h1d, width=0.9, color='#008cf0')
plt.suptitle('Одномерный симметричный цифровой фильтр')
# plt.xlim([-3, 3])
plt.ylim([0, 0.5])
plt.xlabel('pxl')
plt.ylabel('signal')
plt.grid()
plt.show()


""" Формирование функции одномерного цифрового сигнала """
def rect(x, coef=1):
    y = []
    for i in x:
        if np.abs(i - len(x) / 2) <= len(x) / 4 / coef:
            y.append(1.0)
        else:
            y.append(0.0)
    return np.array(y)

M = 180
n = 3
grid_signal = np.arange(int(M / n))
signal = np.concatenate((rect(grid_signal), rect(grid_signal), rect(grid_signal)))
plt.figure()
plt.plot(signal, linewidth=3, color='#008cf0')
plt.suptitle('Цифровой сигнал')
plt.xlim([0, M])
plt.ylim([0, 1.2])
plt.xlabel('pxl')
plt.ylabel('signal')
plt.grid()
plt.show()


""" Фильтрация одномерного сигнала фильтром"""
from scipy.ndimage import convolve
signal_f = convolve(signal, h1d)

plt.figure()
plt.plot(signal_f, linewidth=3, color='#008cf0')
plt.suptitle('Цифровой сигнал')
plt.xlim([0, M])
plt.ylim([0, 1.2])
plt.xlabel('pxl')
plt.ylabel('signal')
plt.grid()
plt.show()


""" Формирование функции двумерного цифрового сигнала """
def img(im_size = 100):
    n_p = int(im_size / a)
    y = np.zeros([M, M])
    y[int(M / 2 - n_p / 2):int(M / 2 + n_p / 2), int(M / 2 - 0.4 * n_p / 2):int(M / 2 + 0.4 * n_p / 2)] = 1
    return y


def add_frame(im, frame_s=100):
    image = np.copy(im)
    n_p = frame_s/2
    image[int(M / 2 - n_p / 2), int(M / 2 - n_p / 2):int(M / 2 + n_p / 2)] = 1
    image[int(M / 2 + n_p / 2), int(M / 2 - n_p / 2):int(M / 2 + n_p / 2)] = 1
    image[int(M / 2 - n_p / 2):int(M / 2 + n_p / 2), int(M / 2 + n_p / 2)] = 1
    image[int(M / 2 - n_p / 2):int(M / 2 + n_p / 2), int(M / 2 - n_p / 2)] = 1
    return image

im = img(im_size=100)
plt.figure()
plt.imshow(add_frame(im), cmap='gray')
plt.suptitle('Исходное изображение')
plt.show()

""" Формирование двумерного симметричного фильтра """
def make_sym(a, dtype=float):
    s = len(a) * 2 - 1
    m = np.zeros([s, s], dtype=dtype)
    m.fill(a[-1])
    for i in range(1, len(a)):
        m[i:s - i, i:s - i].fill(a[-(i + 1)])
    return m

h2d = make_sym(h)
plt.figure()
plt.plot(h2d[2])
plt.suptitle('Изображение после фильтрации')
plt.show()
h2d = h2d / np.sum(h2d)  # Нормировка
print(h2d)
""" Фильтрация одномерного сигнала фильтром"""
im_f = convolve(im, h2d)

plt.figure()
plt.imshow(add_frame(im_f), cmap='gray')
plt.suptitle('Изображение после фильтрации')
plt.show()
