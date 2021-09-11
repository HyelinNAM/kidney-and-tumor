import numpy as np

# case 결과 한 이미지로 취급 -> dice 채점
def mask2dice(true_mask, kidney_mask, tumor_mask):

    '''
    Args:
        true_mask : Array of arbitrary shape.
        pred_mask : Array with the same shape than true_mask.  
    
    Returns:
        A scalar representing the Dice coefficient between the two segmentations.
    
    '''

    assert true_mask.shape == kidney_mask.shape # (512,512) with 0,1,2 value
    assert true_mask.shape == tumor_mask.shape

    true_mask = (np.arange(3) == true_mask[...,None]) # (512,512,3)

    # for kidney
    gt = true_mask[:,:,1:2]
    pred = np.expand_dims(kidney_mask,2) 

    gt = np.asarray(gt).astype(np.bool)
    pred = np.asarray(pred).astype(np.bool)

    # if both are all zero
    im_sum = gt.sum() + pred.sum()

    if im_sum == 0:
        kidney_score = 0

    else:
        intersection = np.logical_and(gt, pred)
        kidney_score = 2 * intersection.sum() / im_sum

    # for tumor
    gt = true_mask[:,:,2:] 
    tumor_mask = np.where(tumor_mask == 2, 1, tumor_mask)
    pred = np.expand_dims(tumor_mask,2)

    gt = np.asarray(gt).astype(np.bool)
    pred = np.asarray(pred).astype(np.bool)

    # if both are all zero
    im_sum = gt.sum() + pred.sum()

    print(gt.sum(), pred.sum(), im_sum)

    if im_sum == 0:
        tumor_score = 0

    else:
        intersection = np.logical_and(gt, pred)
        tumor_score = 2 * intersection.sum() / im_sum

    print(f'kidney {kidney_score:.3f} tumor {tumor_score:.3f}')
    print(f'dice {(kidney_score + tumor_score)/2:.3f}')

    return kidney_score, tumor_score, (kidney_score + tumor_score)/2

# case별 dice로 채점 -> 구현
def case2dice(true_masks, kidney_masks, tumor_masks):

    true_mask = np.vstack(true_masks) # 64,512,512
    kidney_mask = np.vstack(kidney_masks) # 64,512,512
    tumor_mask = np.vstack(tumor_masks) # 64,512,512

    shape = true_mask.shape
    n = int(np.sqrt(shape[0] * shape[1] * shape[2]))

    true_mask = true_mask.reshape(-1,512)
    kidney_mask = kidney_mask.reshape(-1,512)
    tumor_mask = tumor_mask.reshape(-1,512)

    return mask2dice(true_mask, kidney_mask, tumor_mask)