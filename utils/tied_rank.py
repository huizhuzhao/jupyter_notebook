#!/usr/bin/env python
# encoding: utf-8
# Created Time: 2017年02月10日 星期五 22时24分32秒

import numpy as np
import auc
import time

def tied_rank(x):
	sorted_x = sorted(zip(x, range(len(x))))
	r = [0 for k in x]
	cur_val = sorted_x[0][0]
	last_rank = 0

	print 'sorted_x: ', sorted_x
	for i in range(len(sorted_x)):
		if cur_val != sorted_x[i][0]:
			cur_val = sorted_x[i][0]
			for j in range(last_rank, i):
				idx = sorted_x[j][1]
				r[idx] = float(last_rank+1+i)/2.
				
			
			last_rank = i

		if i == len(sorted_x)-1:
			for j in range(last_rank, i+1):
				r[sorted_x[j][1]] = float(last_rank+i+2)/2.
			
def rank(x):
	def get_ave_rank(start_rank, r):
		ave_rank = 0.
		count = 0.
		for idx in range(start_rank, r):
			ave_rank += idx
			count += 1.0	
		ave_rank /= count
		return ave_rank

	total_num = len(x)
	x_sorted = [[ii, r] for ii, r in zip(sorted(x), range(total_num))]
	print x_sorted
	

	start_vari = x_sorted[0][0]
	start_rank = 0
	for v, r in x_sorted:
		if v == start_vari:
			pass

		else:
			ave_rank = get_ave_rank(start_rank, r)
			for idx in range(start_rank, r):
				x_sorted[idx][1] = ave_rank
			start_rank = r
			start_vari = v
		
	ave_rank = get_ave_rank(start_rank, total_num+1)
	
	for idx in range(start_rank, total_num):
		x_sorted[idx][1] = ave_rank

	x_rank_dict = dict(x_r for x_r in x_sorted)

	return [x_rank_dict[ii]+1 for ii in x]



if __name__ == '__main__':
	x = np.arange(10)
	x = [56, 75, 45, 71, 61, 64, 58, 80, 76, 61]
	#x = [1, 2, 2, 2]

	print rank(x)
