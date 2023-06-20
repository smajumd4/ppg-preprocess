def mask_reorient_dummymask(captk_dummy_mask, mask_to_flip):
    """
    adapted from https://stackoverflow.com/questions/66231763/how-to-check-nifti-image-is-in-the-right-orientation-position-with-python
    Flips itk or gandlf masks for use in captk
    Uses the dummy captk mask as a reference
    Check the NIfTI orientation, and flip to  'RPS' if needed.
    :param image: image nifti path
    :param captk_dummy_mask: the dummy captk nifti path
    :param mask_to_flip: the path to the mask to be flipped for use in captk
    :return: mask after flipping
    """
    
    # open itk or gandlf mask
    mask_itk = nib.load(mask_to_flip)
    mask_itk_header = mask_itk.header
    mask_itk_np = np.array(mask_itk.dataobj).astype(np.float64)

    # open captk mask 
    mask_captk = nib.load(captk_dummy_mask)
    mask_captk_header = mask_captk.header

    # print the affine matrices before flipping
    np.set_printoptions(precision=3, suppress=True)
    print("Non-CAPTK Mask affine: ")
    print(mask_itk.affine)
    print("CAPTK affine: ")
    print(mask_captk.affine)

    # flip the axes that do not match
    x, y, z = nib.aff2axcodes(mask_itk.affine)
    if x != nib.aff2axcodes(mask_captk.affine)[0]:
        mask_itk_np = np.flip(mask_itk_np, axis=0)
    if y != nib.aff2axcodes(mask_captk.affine)[1]:
        mask_itk_np = np.flip(mask_itk_np, axis=1)
    if z != nib.aff2axcodes(mask_captk.affine)[2]:
        mask_itk_np = np.flip(mask_itk_np, axis=2)
    
    # create nifti file with the correct orientation
    itk_mask_new = nib.Nifti1Image(mask_itk_np.astype(np.float64), mask_captk.affine, mask_captk.header)
    
    # print the affine matrices after flipping
    print("Non-CAPTK Mask affine after flipping: ")
    print(itk_mask_new.affine)

    return itk_mask_new

def mask_reorient_with_matrices(captk_dummy_mask, mask_to_flip, A_path, ornt_path):
    """
    adapted from https://github.com/nipy/nibabel/issues/1086
    Flips itk or gandlf masks for use in captk
    Uses the affine transformation matrix and orientation array
    Check the NIfTI orientation, and flip to  'RPS' if needed.
    :param image: image nifti path
    :param captk_dummy_mask: the dummy captk nifti path
    :param mask_to_flip: the path to the mask to be flipped for use in captk
    :param A_path: the path to the mask_affine to captk_affine np array
    :param ornt_path: the path to the orientation transform np array
    :return: mask after flipping
    """
    
    with open(A_path, 'rb') as f:
        A = np.load(f)

    with open(ornt_path, 'rb') as f:
        ornt = np.load(f)
    
    # open itk or gandlf mask
    mask_itk = nib.load(mask_to_flip)
    mask_itk_header = mask_itk.header
    mask_itk_np = np.array(mask_itk.dataobj).astype(np.float64)

    # open captk mask 
    mask_captk = nib.load(captk_dummy_mask)
    mask_captk_header = mask_captk.header

    # print the affine matrices before flipping
    np.set_printoptions(precision=3, suppress=True)
    print("Non-CAPTK Mask axes orientation: ")
    print(print(nib.aff2axcodes(mask_itk.affine)))
    print("CAPTK axes orientation: ")
    print(print(nib.aff2axcodes(mask_captk.affine)))

    # flip the axes that do not match
    itk_reorient = mask_itk.as_reoriented(ornt)
    flipped_affine = A @ mask_itk.affine
    
    # create nifti file with the correct orientation
    itk_mask_new = nib.Nifti1Image(itk_reorient.dataobj, flipped_affine, itk_reorient.header)
    
    # print new orientation
    print("Non-CAPTK  axes new orientation: ")
    print(print(nib.aff2axcodes(itk_mask_new.affine)))

    return itk_mask_new