import argparse

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import time
import math

def init_centroids(num_clusters, image):
    """
    Initialize a `num_clusters` x image_shape[-1] nparray to RGB
    values of randomly chosen pixels of `image`

    Parameters
    ----------
    num_clusters : int
        Number of centroids/clusters
    image : nparray
        (H, W, C) image represented as an nparray

    Returns
    -------
    centroids_init : nparray
        Randomly initialized centroids
    """

    # *** START YOUR CODE ***
    random.seed(time.time())
    centroids_init = []
    #m = image.shape[0]
    #print(m)
    for i in range(num_clusters):
        x = random.randrange(image.shape[0])
        y = random.randrange(image.shape[1])
        centroids_init.append(image[x][y])
    centroids_init = np.array(centroids_init)
    print(centroids_init)

    # raise NotImplementedError('init_centroids function not implemented')
    
    # *** END YOUR CODE ***

    return centroids_init


def update_centroids(centroids, image, max_iter=30, print_every=10):
    """
    Carry out k-means centroid update step `max_iter` times

    Parameters
    ----------
    centroids : nparray
        The centroids stored as an nparray
    image : nparray
        (H, W, C) image represented as an nparray
    max_iter : int
        Number of iterations to run
    print_every : int
        Frequency of status update

    Returns
    -------
    new_centroids : nparray
        Updated centroids
    """

    # *** START YOUR CODE ***
    def distances(c0,p0,c1,p1,c2,p2):
        return np.sqrt((int(c0)-int(p0))**2 + (int(c1)-int(p1))**2 + (int(c2)-int(p2))**2)

    def AverageRGB(arr):
        r = 0
        g = 0
        b = 0
        cnt = len(arr)
        for i in range(len(arr)):
            r += arr[i][0]
            g += arr[i][1]
            b += arr[i][2]
        r_avg = r/cnt
        g_avg = g/cnt
        b_avg = b/cnt

        return(r_avg,g_avg,b_avg)


    while max_iter >=0:
        new_centroids = centroids
        clusters = {}
        num_clusters = len(centroids)

        for i in range(num_clusters):
            clusters[i] = []
        #print(clusters)

        for p in image:
            for pix in p:
                dist = []
                for c in centroids:
                    d = distances(c[0], pix[0], c[1], pix[1], c[2], pix[2])
                    dist.append(d)

                dist = np.array(dist)
                #print(dist)
                min_id = np.argmin(dist)
                #print(min_id)
                clusters[min_id].append(pix)
        #print(clusters[0])
        cnt = 0
        while cnt < num_clusters:
            new_centroids[cnt] = AverageRGB(clusters[cnt])
            cnt += 1
        #print(new_centroids)

        max_iter -= 1
        print(max_iter)
        if max_iter%10 == 0:
            print(new_centroids)


    # raise NotImplementedError('update_centroids function not implemented')
        # Usually expected to converge long before `max_iter` iterations
                # Initialize `dist` vector to keep track of distance to every centroid
                # Loop over all centroids and store distances in `dist`
                # Find closest centroid and update `new_centroids`
        # Update `new_centroids`

    # *** END YOUR CODE ***

    return new_centroids


def update_image(image, centroids):
    """
    Update RGB values of pixels in `image` by finding
    the closest among the `centroids`

    Parameters
    ----------
    image : nparray
        (H, W, C) image represented as an nparray
    centroids : int
        The centroids stored as an nparray

    Returns
    -------
    image : nparray
        Updated image
    """

    # *** START YOUR CODE ***
    def distances(c0,p0,c1,p1,c2,p2):
        return np.sqrt((int(c0)-int(p0))**2 + (int(c1)-int(p1))**2 + (int(c2)-int(p2))**2)

    for p in image:
        for pix in p:
            dist = []
            for c in centroids:
                d = distances(c[0], pix[0], c[1], pix[1], c[2], pix[2])
                dist.append(d)
            min_id = np.argmin(dist)
            pix[0] = centroids[min_id][0]
            pix[1] = centroids[min_id][1]
            pix[2] = centroids[min_id][2]

    # raise NotImplementedError('update_image function not implemented')
            # Initialize `dist` vector to keep track of distance to every centroid
            # Loop over all centroids and store distances in `dist`
            # Find closest centroid and update pixel value in `image`
    # *** END YOUR CODE ***

    return image


def main(args):

    # Setup
    max_iter = args.max_iter
    print_every = args.print_every
    image_path_small = args.small_path
    image_path_large = args.large_path
    num_clusters = args.num_clusters
    figure_idx = 0

    # Load small image
    image = np.copy(mpimg.imread(image_path_small))
    print('[INFO] Loaded small image with shape: {}'.format(np.shape(image)))
    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image)
    plt.title('Original small image')
    plt.axis('off')
    savepath = os.path.join('.', 'orig_small.png')
    plt.savefig(savepath, transparent=True, format='png', bbox_inches='tight')

    # Initialize centroids
    print('[INFO] Centroids initialized')
    centroids_init = init_centroids(num_clusters, image)

    # Update centroids
    print(25 * '=')
    print('Updating centroids ...')
    print(25 * '=')
    centroids = update_centroids(centroids_init, image, max_iter, print_every)

    # Load large image
    image = np.copy(mpimg.imread(image_path_large))
    image.setflags(write=1)
    print('[INFO] Loaded large image with shape: {}'.format(np.shape(image)))
    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image)
    plt.title('Original large image')
    plt.axis('off')
    savepath = os.path.join('.', 'orig_large.png')
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')

    # Update large image with centroids calculated on small image
    print(25 * '=')
    print('Updating large image ...')
    print(25 * '=')
    image_clustered = update_image(image, centroids)

    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image_clustered)
    plt.title('Updated large image')
    plt.axis('off')
    savepath = os.path.join('.', 'updated_large.png')
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')

    print('\nCOMPLETE')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--small_path', default='./peppers-small.tiff',
                        help='Path to small image')
    parser.add_argument('--large_path', default='./peppers-large.tiff',
                        help='Path to large image')
    parser.add_argument('--max_iter', type=int, default=250,
                        help='Maximum number of iterations')
    parser.add_argument('--num_clusters', type=int, default=16,
                        help='Number of centroids/clusters')
    parser.add_argument('--print_every', type=int, default=10,
                        help='Iteration print frequency')
    args = parser.parse_args()
    main(args)
