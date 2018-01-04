import sys
import numpy as np
import pydensecrf.densecrf as dcrf

from pydensecrf.utils import compute_unary, create_pairwise_bilateral, \
    create_pairwise_gaussian, unary_from_softmax

import skimage.io as io

def im_label_crf(image_path,softmax_prob):

    #add image path as argument here
    image = image_path

    #give softmax probabilities as argument

    #softmax = final_probabilities.squeeze()

    #softmax = processed_probabilities.transpose((2, 0, 1))

    # The input should be the negative of the logarithm of probability values
    # Look up the definition of the softmax_to_unary for more information
    unary = unary_from_softmax(softmax_prob)


    #figure out a way to add partial gt information here
    #one trick could be making prob 1 of those of gts in softmax for GT labels

    # The inputs should be C-continious -- we are using Cython wrapper
    unary = np.ascontiguousarray(unary)

    d = dcrf.DenseCRF(image.shape[0] * image.shape[1], 19)

    d.setUnaryEnergy(unary)

    # This potential penalizes small pieces of segmentation that are
    # spatially isolated -- enforces more spatially consistent segmentations
    feats = create_pairwise_gaussian(sdims=(10, 10), shape=image.shape[:2])

    d.addPairwiseEnergy(feats, compat=3,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This creates the color-dependent features --
    # because the segmentation that we get from CNN are too coarse
    # and we can use local color features to refine them
    feats = create_pairwise_bilateral(sdims=(50, 50), schan=(20, 20, 20),
                                       img=image, chdim=2)

    d.addPairwiseEnergy(feats, compat=10,
                         kernel=dcrf.DIAG_KERNEL,
                         normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(5)

    res = np.argmax(Q, axis=0).reshape((image.shape[0], image.shape[1]))

#cmap = plt.get_cmap('bwr')

#f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
#ax1.imshow(res, vmax=1.5, vmin=-0.4, cmap=cmap)
#ax1.set_title('Segmentation with CRF post-processing')
#probability_graph = ax2.imshow(np.dstack((train_annotation,)*3)*100)
#ax2.set_title('Ground-Truth Annotation')
#plt.show()
