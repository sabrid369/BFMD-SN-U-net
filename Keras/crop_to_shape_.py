

def crop_to_shape(data, shape):

    offset0 = (data.shape[1] - shape[1])//2
    offset1 = (data.shape[2] - shape[2])//2
    if offset0==0:
        if data.shape[1] % 2 == 1 or shape[1] % 2 == 1:
            return data[:, offset0:data.shape[1], offset1:(-offset1)]
        elif data.shape[2] % 2 == 1 or shape[2] % 2 == 1:
            return data[:, offset0:data.shape[1], offset1:(-offset1 - 1)]
        else:
            return data[:, offset0:data.shape[1], offset1:(-offset1)]
    elif offset1==0:
        if data.shape[1] % 2 == 1 or shape[1] % 2 == 1:
            return data[:, offset0:(-offset0 - 1), offset1:data.shape[2]]
        elif data.shape[2] % 2 == 1 or shape[2] % 2 == 1:
            return data[:, offset0:-offset0, offset1:data.shape[2]]
        else:
            return data[:, offset0:-offset0, offset1:data.shape[2]]
    else:
        if data.shape[1] % 2 == 1 or shape[1] % 2 == 1:
            return data[:, offset0:(-offset0 - 1), offset1:(-offset1)]
        elif data.shape[2] % 2 == 1 or shape[2] % 2 == 1:
            return data[:, offset0:-offset0, offset1:(-offset1-1)]
        else:
            return data[:, offset0:-offset0, offset1:(-offset1)]
