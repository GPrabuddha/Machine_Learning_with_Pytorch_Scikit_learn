import numpy as np


a = [1., 2., 3.]
nd_arr = np.array(a)
print(nd_arr)

two_d_lst = [[1, 2, 3],[4, 5, 6]]
two_d_arr = np.array(two_d_lst)
print(two_d_arr)
print(two_d_arr[0][1])   
two_d_arr[0][1] = 5      # we can change the elements of the array like list

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


zero_arr = np.zeros((3, 3)) + 55
print(zero_arr)
one_arr = np.ones((2, 3)) * 11
print(one_arr)

random_arr = np.empty((1, 3)) 
random_arr_2 = np.empty([1, 2])
print(random_arr)
print(random_arr_2)

identity_matrix = np.eye(5)
print(identity_matrix)

diag_matrix = np.diag([1, 2, 3])
print(diag_matrix)

range_list = np.arange(1., 9, 1) # range can't create float ranges only integer is allowed
print(range_list)

lin_range = np.linspace(1., 15., num= 14) # unlike arange and range, linspace include the end range
print(lin_range)

