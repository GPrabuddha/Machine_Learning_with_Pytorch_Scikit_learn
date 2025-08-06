import numpy as np


a = [1., 2., 3.]
nd_arr = np.array(a)
print(nd_arr)

two_d_lst = [[1, 2, 3],[4, 5, 6]]
two_d_arr = np.array(two_d_lst)
print(two_d_arr)
print(two_d_arr[0][1])
print(two_d_arr.shape)
print(two_d_arr.shape[0])
print(two_d_arr.shape[1])

print(two_d_arr.dtype)
two_d_arr_int16 = two_d_arr.astype(np.int16)
print(two_d_arr_int16.dtype)
two_d_arr_float32 = np.array([1.2, 6.2, 6.3], dtype=np.float16)
print(two_d_arr_float32.dtype)

print(two_d_arr.size)
print(two_d_arr.ndim)