import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

M = 180  # Размер матричного ПИ
a = 2  # мкм, Размер элемента ПИ
n = 3  # Порядокфильтра

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
nu_N = 1 / (2 * a / 1000)
print('Частота Найквиста = %0.2f 1/mm' % nu_N)

""" Нормированная аналоговая функция фильтра"""
grid_norm = np.arange(0, 1, 1 / nu_N)
H1 = H(grid_norm, nu_N)

""" Дискретные значения"""
grid_d = np.arange(0, 0.7, 0.3)
H_d = H(grid_d, nu_N)
print('Дискретные отсчеты: \n', grid_d, '\n Матрица разложения: \n', H_d)

plt.figure()
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


def my_conv(f, kernel):
    pad = int(len(kernel)/2-1/2)
    if len(f.shape)==1:
        filt = np.zeros(f.size)
        f = np.pad(f, [pad, pad], mode='edge')
        for i in range(pad, len(f) - pad):
            for j in range(len(kernel)):
                filt[i - pad] += kernel[j] * f[i - int(len(kernel) / 2 - 1 / 2) + j]
    else:
        filt = np.zeros(f.shape)
        f = np.pad(f, ([pad, pad], [pad, pad]), mode='edge')
        for i in range(pad, len(f) - pad):
            for i2 in range(pad, len(f) - pad):
                for j in range(len(kernel)):
                    for j2 in range(len(kernel)):
                        filt[i-pad, i2-pad] += kernel[j, j2] * f[i - int(len(kernel) / 2 - 1 / 2)+j,
                                                                 i2 - int(len(kernel) / 2 - 1 / 2) + j2]
    return filt

s_f = my_conv(signal, h1d)
""" Фильтрация одномерного сигнала фильтром"""
signal_f = convolve(signal, h1d)
plt.figure()
# plt.plot(signal_f, linewidth=3, color='#008cf0')
plt.plot(s_f, linewidth=3, color='#008cf0')
plt.plot(signal, linewidth=1, color='#000000')
plt.suptitle('Цифровой сигнал')
plt.xlim([0, M])
plt.ylim([0, 1.2])
plt.xlabel('pxl')
plt.ylabel('signal')
plt.grid()
plt.show()

""" Формирование функции двумерного цифрового сигнала """


def img(im_size=180):
    n_p = int(im_size / a)
    y = np.zeros([M, M])
    y[int(M / 2 - n_p / 2):int(M / 2 + n_p / 2), int(M / 2 - 0.4 * n_p / 2):int(M / 2 + 0.4 * n_p / 2)] = 1
    return y


def add_frame(im, frame_s=180):
    image = np.copy(im)
    n_p = frame_s / 2
    image[int(M / 2 - n_p / 2), int(M / 2 - n_p / 2):int(M / 2 + n_p / 2)] = 1
    image[int(M / 2 + n_p / 2), int(M / 2 - n_p / 2):int(M / 2 + n_p / 2)] = 1
    image[int(M / 2 - n_p / 2):int(M / 2 + n_p / 2), int(M / 2 + n_p / 2)] = 1
    image[int(M / 2 - n_p / 2):int(M / 2 + n_p / 2), int(M / 2 - n_p / 2)] = 1
    return image


im = img(im_size=360)
plt.figure()
plt.imshow(im, cmap='gray')
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
im2 = my_conv(im,h2d)
plt.figure()
plt.imshow(im2, cmap='gray')
plt.suptitle('Изображение после фильтрации')
plt.show()
