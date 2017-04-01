import os
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt

class BucketData(object):
	def __init__(self):
		self.max_width=0
		self.max_label_len = 0
		self.data_list = []
		self.label_list = []
		self.file_list = []

	def append(self, datum, label, filename):
		self.data_list.append(datum)
		self.label_list.append(label)
		self.file_list.append(filename)
		self.max_width = max(datum.shape[-1], self.max_width)
		self.max_label_len = max(len(label), self.max_label_len)
		return len(self.data_list)

	def flush_out(self,bucket_specs,valid_target_length=float('inf'),go_shift=1):
		res = {'bucket_id':None,'data':None,'zero_paddings':None,'encoder_mask':None,'decoder_inputs':None,'target_weights':None,'enc_pad_len':None,'decoder_expectation':None}
		
		def get_bucket_id():
			for i in range(0,len(bucket_specs)):
				if bucket_specs[i][0] >= self.max_width/8 -1 and bucket_specs[i][1] >= self.max_label_len:
					return i
			return None

		res['bucket_id'] = get_bucket_id()
		
		if res['bucket_id'] is None:
			self.data_list,self.label_list=[],[]
			self.max_width,self.max_label_len=0,0
			return None

		enc_ip_len , dec_ip_len = bucket_specs[res['bucket_id']]
		
		
		res['data'] = np.array(self.data_list)
		real_len = max(self.max_width/8 - 1,0)
		pad_len = enc_ip_len - real_len

		res['enc_pad_len'] = pad_len
		res['zero_paddings'] = np.zeros([len(self.data_list),pad_len,512],dtype=np.float32)
		encoder_mask = np.concatenate((np.ones([len(self.data_list),real_len]),np.zeros([len(self.data_list),pad_len])),axis=1)
		
		res['encoder_mask']=[a[:,np.newaxis] for a in encoder_mask.T]
		res['real_len'] = self.max_width

		decoder_expectation=[]
		for i in range(len(self.label_list)):
			exp_op = self.label_list[i][1:-1]
			pad_end_tag_len = dec_ip_len - len(exp_op)
			if pad_end_tag_len >= 0 :
				decoder_expectation.append(np.concatenate((np.array(exp_op),np.ones(pad_end_tag_len,dtype=np.int32)*2),axis=0))
			else:
				raise NotImplementedError

		target_weights=[]
		
		for i in range(len(self.label_list)):
			label_len = len(self.label_list[i])
			if label_len <= dec_ip_len:
				self.label_list[i] = np.concatenate((self.label_list[i],np.zeros(dec_ip_len-label_len,dtype=np.int32)))
				one_mask_len = min(label_len-go_shift,valid_target_length)
				target_weights.append(np.concatenate((np.ones(one_mask_len,dtype=np.float32),np.zeros(dec_ip_len-one_mask_len,dtype=np.float32))))
			else:
				raise NotImplementedError
		

		res['decoder_expectation'] = [a.astype(np.int32) for a in np.array(decoder_expectation)]

		
		res['decoder_inputs']=[a.astype(np.int32) for a in np.array(self.label_list).T]
		res['target_weights']=[a.astype(np.int32) for a in np.array(target_weights).T]

		assert len(res['decoder_inputs']) == len(res['target_weights'])
		res['filenames'] = self.file_list

		self.data_list, self.label_list, self.file_list = [],[],[]
		self.max_width,self.max_label_len=0,0

		return res

	def __len__(self):
		return len(self.data_list)
	
	def __iadd__(self, other):
		self.data_list += other.data_list
		self.label_list += other.label_list
		self.max_label_len = max(self.max_label_len, other.max_label_len)
		self.max_width = max(self.max_width, other.max_width)
	
	def __add__(self,other):
		res = BucketData()
		res.data_list = self.data_list + other.data_list
		res.label_list = self.label_list + other.label_list
		res.max_width = max(self.max_width, other.max_width)
		res.max_label_len = max((self.max_label_len, other.max_label_len))
		return res


class DataGen(object):
	def __init__(self,data_root,image_dir,formula_lst,data_lst,mapv,mapi,evaluate=False,valid_target_length=float('inf'),img_width_range=(120,400),word_len=160): #

		img_h = 50;
		self.GO = 1
		self.EOS = 2
		
		self.data_root = data_root
		
		if os.path.exists(image_dir):
			self.image_path = image_dir
		else:
			self.image_path = os.path.join(data_root,image_dir)

		if os.path.exists(formula_lst):
			self.form_map = formula_lst
		else:
			self.form_map = os.path.join(data_root,formula_lst)

		if os.path.exists(formula_lst):
			self.data_map = data_lst
		else:
			self.data_map = os.path.join(data_root,data_lst)

		if evaluate:
			self.bucket_specs = [(160/8 -1, word_len+ 2), (200/8 - 1, word_len + 2),(240/8 - 1, word_len + 2), (280/8 -1, word_len + 2),(320/8 -1, word_len + 2),(360/8 -1, word_len + 2),(400/8-1, word_len + 2)]
		else:
			self.bucket_specs = [(160/8 -1, 25+ 2), (200/8 -1, 50 + 2),(240/8 -1, 75 + 2), (280/8 -1, 100 + 2),(320/8 -1, 125 + 2),(360/8 -1, 150 + 2),(400/8 -1, 175+2)]
		
		if evaluate:
			self.bucket_specs = [(160/8 -1, 150+ 2), (200/8 - 1, 150 + 2),(240/8 - 1, 150 + 2), (280/8 -1, 150 + 2),(320/8 -1, 150 + 2),(360/8 -1, 150 + 2),(400/8-1, 150 + 2)]
		else:
			self.bucket_specs = [(160/8 -1, 25+ 2), (200/8 -1, 50 + 2),(240/8 -1, 75 + 2), (280/8 -1, 100 + 2),(320/8 -1, 125 + 2),(360/8 -1, 150 + 2),(400/8 -1, 175+2)]


		self.bucket_min_width, self.bucket_max_width = img_width_range
		self.image_height = img_h
		self.valid_target_len = valid_target_length
		self.bucket_data = {i: BucketData() for i in range(self.bucket_max_width + 1)}

		d_map = open(self.data_map,'rb').readlines()
		f_map = open(self.form_map,'rb').readlines()
		self.d_map_dict={}
		self.f_map_dict = {}
		for line in d_map:
			im_name, tag = line.strip().split()
			self.d_map_dict[im_name] = tag

		 
		self.map_v=mapv
		self.map_i=mapi
		it=0
		for form in f_map:
			self.f_map_dict[it] = form
			it += 1

		


	def clear(self):
		print "Clearing"
		self.bucket_data = {i: BucketData() for i in range(self.bucket_max_width + 1)}

	def gen(self,batch_size):
		valid_target_length = self.valid_target_len
		image_list = os.listdir(self.image_path)
		random.shuffle(image_list)
		
		for im in image_list:
			if im in self.d_map_dict:
				img_path,lex = self.image_path+'/'+im,self.f_map_dict[(int)(self.d_map_dict[im])].strip().split()
				if len(lex) == 0 or len(lex) >= self.bucket_specs[-1][1]:
					print img_path
					continue
				try:
					img_bw, target = self.read_data(img_path, lex)
					if valid_target_length < float('inf'):
						target = target[:valid_target_len + 1]
					width = img_bw.shape[-1]
					
					b_idx = min(width, self.bucket_max_width)
					bs = self.bucket_data[b_idx].append(img_bw, target, os.path.join(self.data_root,img_path)) ## Very Very Important. You need to collect images of same width size together
					
					if bs >= batch_size:
						b = self.bucket_data[b_idx].flush_out(self.bucket_specs,valid_target_length=valid_target_length,go_shift=1)
						if b is not None:
							yield b
						else:
							assert False, 'no valid bucket of width %d'%width
				except IOError:
					pass

		self.clear()

	

	def read_data(self, img_path, lex):
		assert 0 < len(lex) < self.bucket_specs[-1][1]
		with open(img_path, 'rb') as img_file:
			img = Image.open(img_file)
			w, h = img.size
			
			aspect_ratio = float(w) / float(h)
			if aspect_ratio < float(self.bucket_min_width) / self.image_height:
				img = img.resize((self.bucket_min_width, self.image_height),Image.ANTIALIAS)
			elif aspect_ratio > float(self.bucket_max_width) / self.image_height:
				img = img.resize((self.bucket_max_width, self.image_height),Image.ANTIALIAS)
			elif h != self.image_height:
				img = img.resize((int(aspect_ratio * self.image_height), self.image_height),Image.ANTIALIAS)
			
			img_bw = img.convert('L')
			img_bw = np.asarray(img_bw, dtype=np.uint8)
			img_bw = img_bw[np.newaxis,:]
			
			
			formula = [self.GO]
			for token in lex:
				if token in self.map_v:
					formula.append(self.map_v[token])
				else:
					formula.append(3)
			formula.append(self.EOS)
			formula = np.array(formula, dtype=np.int32)

		return img_bw, formula



def test_datagen(root,img,f_lst,d_lst):
	map_v={}
	map_i={}
	f_lines = open(root+'/'+f_lst,'rb').readlines()
	tag=3
	total_images = os.listdir(root+'/'+img)
	total_train = open(root+'/'+d_lst,'rb').readlines()
	print len(total_train)
	print len(total_images)
	for line in f_lines:
		tokens = line.strip().split()
		for token in tokens:
			if token not in map_v:
				map_v[token] = tag
				map_i[tag] = token
				tag += 1
	print len(map_v.values())
	obj = DataGen(root,img,f_lst,d_lst,map_v,map_i,valid_target_length=float('inf'),evaluate=False)
	count = 0 
	for xxx in range(10):
		for batch in obj.gen(50):
			#print "Iteration\t{}".format(count)
			count += 1
		print count
		
#res = {'bucket_id':None,'data':None,'zero_paddings':None,'encoder_mask':None,'decoder_inputs':None,'target_weights':None}
#test_datagen('/home/fallsrisk/Documents/ishan/tensorflow/sample','images_processed','formulas.norm.lst','train_filter.lst')













