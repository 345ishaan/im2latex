import os
import numpy as np
import random
from PIL import Image


class BucketData(object):
	def __init__(self):
		# self.max_width=0
		# self.max_label_len = 0
		self.data_list = []
		self.label_list = []
		#self.file_list = []

	def append(self, datum, label, filename):
		self.data_list.append(datum)
		self.label_list.append(label)
		return len(self.data_list)

	def flush_out(self,bucket_specs,valid_target_length=float('inf'),go_shift=1):
		res = {'decoder_inputs':None,'data':None,'decoder_expectation':None}
		
		
		
		res['data'] = np.array(self.data_list)
		res['decoder_inputs']=[a.astype(np.int32) for a in np.array(self.label_list).T]

		
		
		decoder_expectation=[]
		for i in range(len(self.label_list)):
			exp_op = self.label_list[i][1:]
			pad_end_tag_len = 1
			if pad_end_tag_len >= 0 :
				decoder_expectation.append(np.concatenate((np.array(exp_op),np.zeros(pad_end_tag_len,dtype=np.int32))))
			else:
				raise NotImplementedError

		res['decoder_expectation'] = [a.astype(np.int32) for a in np.array(decoder_expectation)]

		
		
		self.data_list, self.label_list = [],[] 
		
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
		self.PAD =0
		self.MAXWIDTH = 400
		self.MAXHEIGHT = 160
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
		## Find different variant of widht and heights available
		#Analyse the number of latex symbols in a particular range of width and height
		#Set the corresponding number as the bucket size
		
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

		self.size_dict = self.find_sizes()

		self.bucket_data = {key : BucketData() for key in self.size_dict}




	def find_sizes(self):
		size_dict = {}
		for key in self.d_map_dict.keys():
			img = Image.open(self.image_path+'/'+key)
			w,h = img.size
			if w > self.MAXWIDTH and h < self.MAXHEIGHT:
				aspect_ratio = float(w) / float(h)
				new_w = self.MAXWIDTH
				new_h = (int(new_w/aspect_ratio))
				img = img.resize((new_w,new_h),Image.ANTIALIAS)
			elif w < self.MAXWIDTH and h > self.MAXHEIGHT:
				aspect_ratio = float(w) / float(h)
				new_h = self.MAXHEIGHT
				new_w = (int(new_h *aspect_ratio))
				img = img.resize((new_w,new_h),Image.ANTIALIAS)
			elif w > self.MAXWIDTH and h > self.MAXHEIGHT:
				img = img.resize((self.MAXWIDTH,self.MAXHEIGHT),Image.ANTIALIAS)

			if img.size in size_dict:
				size_dict[img.size]['name'].append(key)
			else:
				size_dict.setdefault(img.size,{'name':[],'len':0})
			f_len = len(self.f_map_dict[(int)(self.d_map_dict[key])].strip().split())
			if f_len+2 > size_dict[img.size]['len']:
				size_dict[img.size]['len'] = f_len+2
			
		return size_dict

		
	def clear(self):
		print "Clearing"
		self.bucket_data = {key : BucketData() for key in self.size_dict}		

	# def clear(self):
	# 	print "Clearing"
	# 	self.bucket_data = {i: BucketData() for i in range(self.bucket_max_width + 1)}

	
	def gen(self,batch_size):
		for key in  self.size_dict:
			list_img = self.size_dict[key]['name']
			bucket_size = self.size_dict[key]['len']
			for im in list_img:
				img_path,lex = self.image_path+'/'+im,self.f_map_dict[(int)(self.d_map_dict[im])].strip().split()
				if len(lex) == 0 or len(lex) >= 200:
					print img_path
					continue
				try:
					img_bw, target = self.read_data(img_path, lex,bucket_size)
					wi = img_bw.shape[-1]
					he = img_bw.shape[-2]

					bs = self.bucket_data[(wi,he)].append(img_bw, target, os.path.join(self.data_root,img_path)) ## Very Very Important. You need to collect images of same width size together
					
					if bs >= batch_size:
						b = self.bucket_data[(wi,he)].flush_out((wi,he),go_shift=1)
						if b is not None:
							yield b
						else:
							assert False, 'no valid bucket of width %d'%width
				except IOError:
					pass

		self.clear()

	def read_data(self, img_path, lex,bucket_size):
		
		with open(img_path, 'rb') as img_file:
			img = Image.open(img_file)
			w, h = img.size
			
			
			if w > self.MAXWIDTH and h < self.MAXHEIGHT:
				aspect_ratio = float(w) / float(h)
				new_w = self.MAXWIDTH
				new_h = (int(new_w/aspect_ratio))
				img = img.resize((new_w,new_h),Image.ANTIALIAS)
			elif w < self.MAXWIDTH and h > self.MAXHEIGHT:
				aspect_ratio = float(w) / float(h)
				new_h = self.MAXHEIGHT
				new_w = (int(new_h *aspect_ratio))
				img = img.resize((new_w,new_h),Image.ANTIALIAS)
			elif w > self.MAXWIDTH and h > self.MAXHEIGHT:
				img = img.resize((self.MAXWIDTH,self.MAXHEIGHT),Image.ANTIALIAS)
			

			
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
			for ii in range(bucket_size-len(formula)):
				formula.append(self.PAD)
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
	 	for batch in obj.gen(20):
	 		print np.array(batch['data']).shape
	 		print np.array(batch['decoder_inputs']).shape
	 		print np.array(batch['decoder_expectation'])[0]
	 		print np.array(batch['decoder_inputs']).T[0]
	 		break

	 		#print "Iteration\t{}".format(count)
	 		count += 1
	 	break
	 	print count
		
#res = {'bucket_id':None,'data':None,'zero_paddings':None,'encoder_mask':None,'decoder_inputs':None,'target_weights':None}
#test_datagen('/home/fallsrisk/Documents/ishan/tensorflow/im2latex_dataset/datafull/sample/data','smaller_images_processed','formulas.norm.lst','train_filter.lst')













