"""
This file contains the code for shared memeory utility functions
"""
__author__ = "Dinusha Nuwan"
__copyright__ = "Copyright 2022, pAIx"
__version__ = "0.0.1"
__maintainer__ = "Nuwan1654"
__email__ = "dinusha@zoomi.ca"
__status__ = "Staging"


import numpy as np
import copy
from multiprocessing import shared_memory
import pickle
import cv2
"""
Data types

data['frameNumber'] - int
data['frameTime']   - time.monotonic()
data['frame']       - np.ndarray
data['frameRate']   - int 
data['dest']        - string 
data['cameraID']    - int 

int	    4 bytes
long	4 bytes
float	4 bytes
double	8 bytes
"""

def utf8len(s):
    return len(s.encode('utf-8'))

def sm_write_frame(image):
    h, w, c = image.shape
    shm_ = shared_memory.SharedMemory(create=True, size=h*w*c)
    # np array backed by shared memory
    im = np.ndarray((h,w,c), dtype=np.uint8, buffer=shm_.buf) 
    # copy data to array
    im[:] = image 
    ret = shm_.name
    # closes access to the shared memory from this instance
    shm_.close() 
    return ret, (h,w,c)

def sm_read_frame(sm_name, shape, unlink=True):
    shm_ = shared_memory.SharedMemory(name=sm_name)
    # im is backed by shared memory
    im = np.ndarray(shape, dtype=np.uint8, buffer=shm_.buf)
    # retun array to copy the sared array
    im_ = np.ndarray((shape), dtype=np.uint8) 
    im_[:] = im[:]
    if unlink:
        shm_.close()
        shm_.unlink()
    else:
        shm_.close()
    return im_

def sm_write_int(integer_val):
    sml = shared_memory.ShareableList([integer_val])
    ret = sml.shm.name
    sml.shm.close()
    return ret

def sm_read_int(sm_name, unlink=True):
    shm_ = shared_memory.ShareableList(name=sm_name)
    ret = copy.copy(shm_[0])
    if unlink:
        shm_.shm.close()
        shm_.shm.unlink()
    else:
        shm_.shm.close()
    return ret


def sm_write_float(float_val):
    sml = shared_memory.ShareableList([float_val])
    ret = sml.shm.name
    sml.shm.close()
    return ret

def sm_read_float(sm_name, unlink=True):
    shm_ = shared_memory.ShareableList(name=sm_name)
    ret = copy.copy(shm_[0])
    if unlink:
        shm_.shm.close()
        shm_.shm.unlink()
    else:
        shm_.shm.close()
    return ret

def sm_write_string(string_val):
    sml = shared_memory.ShareableList([string_val])
    ret = sml.shm.name
    sml.shm.close()
    return ret

def sm_read_string(sm_name, unlink=True):
    shm_ = shared_memory.ShareableList(name=sm_name)
    ret = copy.copy(shm_[0])
    if unlink:
        shm_.shm.close()
        shm_.shm.unlink()
    else:
        shm_.shm.close()
    return ret


def sm_write_string_list(string_list):
    sml = shared_memory.ShareableList(string_list)
    ret = sml.shm.name
    sml.shm.close()
    return ret, len(string_list)

def sm_read_string_list(sm_name, length, unlink=True):
    ret = []
    shm_ = shared_memory.ShareableList(name=sm_name)
    for i in range(length):
        ret.append(shm_[i])
    if unlink:
        shm_.shm.close()
        shm_.shm.unlink()
    else:
        shm_.shm.close()
    return ret

def sm_write_nparray(array_):
    shm_ = shared_memory.SharedMemory(create=True, size=array_.nbytes)
    b = np.ndarray(array_.shape, dtype=array_.dtype, buffer=shm_.buf) 
    b[:] = array_[:]
    ret = shm_.name
    shm_.close()
    return ret, array_.shape, array_.dtype

def sm_read_nparray(sm_name, shape, dtype, unlink=True):
    shm_ = shared_memory.SharedMemory(name=sm_name)
    # ret is backed by shared memory
    ret = np.ndarray((shape), dtype=dtype, buffer=shm_.buf) 
    # retun array to copy the sared array
    arr_ = np.ndarray((shape), dtype=dtype) 
    arr_[:] = ret[:]
    if unlink:
        shm_.close()
        shm_.unlink()
    else:
        shm_.close()
    return arr_

def pickle_dict(data):
    pobj = pickle.dumps(data, -1)
    size_ = len(pobj)
    shm_ = shared_memory.SharedMemory(create=True, size=size_)
    shm_.buf[:] = pobj
    ret = shm_.name
    shm_.close()
    return ret

def unpickle_dict(sm_name, unlink=True):
    shm_ = shared_memory.SharedMemory(name=sm_name)
    data = pickle.loads(shm_.buf)
    if unlink:
        shm_.close()
        shm_.unlink()
    else:
        shm_.close()
    return data



    

if __name__ == '__main__':
    # Test pickle
    data = {
        'a' : np.arange(30, dtype=np.int64),
        'b' : np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]]),
        'c' : 12.345,
        'd' : 'fds',
        'e' : 1,
        'f' : [[1,2,3] ,[4,5,6,7]]
    }

    sm_name = pickle_dict(data)
    ret = unpickle_dict(sm_name)
    print(ret)
    print(ret['d'])

    # Test np arrays
    arr = np.array([1.7, 2.9])
    print(arr.shape, arr.dtype)
    sm_name, shape, dtype =  sm_write_nparray(arr)
    ret = sm_read_nparray(sm_name, shape, dtype)
    print(ret)

    # Test image
    img = cv2.imread('/home/dinusha/pAIx-shared-mem/pAIx/16.jpg') 
    sm_name, shape = sm_write_frame(img)
    ret = sm_read_frame(sm_name, shape)
    cv2.imwrite('test.jpg', ret)

    # Test int
    shm_ = sm_write_int(26)
    ret = sm_read_int(shm_)
    print(ret)

    # Test String
    shm_ = sm_write_string('ffdddd')
    ret = sm_read_string(shm_)
    print(ret)

    # Test String list
    shm_, lenth = sm_write_string_list(['asd', 'sddf', 'ffrrf'])
    ret = sm_read_string_list(shm_, lenth)
    print(ret)

    # Test float
    shm_ = sm_write_float(234.567)
    ret = sm_read_float(shm_)
    print(ret)
