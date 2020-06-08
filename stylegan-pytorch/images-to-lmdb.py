import argparse
from io import BytesIO
import multiprocessing
from functools import partial

from PIL import Image
import lmdb
from tqdm import tqdm
from torchvision import datasets
from torchvision.transforms import functional as trans_fn


def resize_and_convert(img, size, quality=100):
    img = trans_fn.resize(img, size, Image.LANCZOS)
    img = trans_fn.center_crop(img, size)
    #instead of saving to a dir it saves to RAM
    buffer = BytesIO()
    img.save(buffer, format='jpeg', quality=quality)
    val = buffer.getvalue()

    return val

#512 and 1024 sizes removed
def resize_multiple(img, sizes=(8, 16, 32, 64, 128, 256), quality=100):
    imgs = []

    for size in sizes:
        imgs.append(resize_and_convert(img, size, quality))

    return imgs


def resize_worker(img_file, sizes):
    i, file = img_file
    img = Image.open(file)
    img = img.convert('RGB')
    out = resize_multiple(img, sizes=sizes)

    return i, out


def prepare(env, dataset, n_worker, sizes=(8, 16, 32, 64, 128, 256)):
    #partial function whose argument size is always substituted with the given parameter
    #so resize_fn behaves like resize_worker but takes in 1 less argument
    
    resize_fn = partial(resize_worker, sizes=sizes)

    #lambda function is anonymous. so here it sorts according to the first element of input 
    files = sorted(dataset.imgs, key=lambda x: x[0])
    #add index and remove labels
    files = [(i, file) for i, (file, label) in enumerate(files)]

    total = 0

    with multiprocessing.Pool(n_worker) as pool:
    	#imap_unordered smooth things out by yielding faster-calculated values 
    	#ahead of slower-calculated values so it does not mean faster execution
    	#just that you can see the results earlier

        for i, imgs in tqdm(pool.imap_unordered(resize_fn, files)):
            for size, img in zip(sizes, imgs):
            	#zfill pad 0s on the left
                key = f'{size}-{str(i).zfill(5)}'.encode('utf-8')
                #store a record
                with env.begin(write = True) as txn:
                    txn.put(key, img)
            total += 1

        print(total)
        #store a record
        with env.begin(write = True) as txn:
            txn.put('length'.encode('utf-8'), str(total).encode('utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str)
    parser.add_argument('--n_worker', type=int, default=1)
    parser.add_argument('path', type=str)

    args = parser.parse_args()

    imgset = datasets.ImageFolder(args.path)

    #map size determines the size of mdb file, it is fixed
    with lmdb.open(args.out, map_size = 2*1024**3, readahead=False) as env:
        prepare(env, imgset, args.n_worker)


# what does this file do? is it like reading fro and writing to a txt file? 