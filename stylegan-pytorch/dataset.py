from io import BytesIO
import torch
import lmdb
from PIL import Image
from torch.utils.data import Dataset

class MultiResolutionDataset(Dataset):
	'''
	a custom dataset
	but the multiresoluton conversion has been done in images-to-lmdb.py
	'''

	def __init__(self, path, transform, resolution=8):
		self.env = lmdb.open(
			path,
			max_readers=32,
			readonly=True,
			lock=False,
			readahead=False,
			meminit=False,
		)

		if not self.env:
			raise IOError('Cannot open lmdb dataset', path)
		
		with self.env.begin(write=False) as txn:
			self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))
		
		self.resolution = resolution
		self.transform = transform

	def __len__(self):
		return self.length

	def __getitem__(self, index):
		with self.env.begin(write=False) as txn:
			key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
			img_bytes = txn.get(key)

		buffer = BytesIO(img_bytes)
		img = Image.open(buffer)
		img = self.transform(img)

		return img

class TwoClassDataset(Dataset):
	def __init__(self,original,generated,size):
		tuple_original = tuple([original[i][0].unsqueeze(0) for i in range(size)])
		tuple_generated = tuple([generated[i][0].unsqueeze(0) for i in range(size)])
		#concat all individual tensors in the tuple to one tensor
		tensor_original = torch.cat(tuple_original,0)
		tensor_generated = torch.cat(tuple_generated,0)
		
		self.data = torch.cat(tuple([tensor_original,tensor_generated]),0)

		self.labels = torch.cat(tuple([torch.ones([size]),torch.zeros([size])]))

	def __len__(self):
		return self.data.size()[0]

	def __getitem__(self,index):
		data,labels = self.data[index],self.labels[index]

		return data, labels




