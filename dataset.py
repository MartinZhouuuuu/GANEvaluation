from io import BytesIO
import torch
import lmdb
from PIL import Image
from torch.utils.data import Dataset

class MultiResolutionDataset(Dataset):
	'''
	a custom dataset
	but the multiresoluton conversion has been done in prepare_data.py
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
	def __init__(self,original,generated):
		tuple_oringal = tuple([original[i].unsqueeze(0) for i in range(100)])
		tuple_generated = tuple([generated[i][0].unsqueeze(0) for i in range(100)])
		tensor_original = torch.cat(tuple_oringal,0)
		tensor_generated = torch.cat(tuple_generated,0)
		
		self.data = torch.cat(tuple([tensor_original,tensor_generated]),0)

		self.labels = torch.cat(tuple([torch.ones([100],dtype = torch.int8),torch.zeros([100],dtype = torch.int8)]))

	def __len__(self):
		return self.data.size()[0]

	def __getitem__(self,index):
		data,labels = self.data[index],int(self.labels[index])

		return data, labels




