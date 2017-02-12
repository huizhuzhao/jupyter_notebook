#!/usr/bin/env python
# encoding: utf-8
# Created Time: 2017年02月10日 星期五 22时24分32秒

import numpy as np
import time

def itied_rank(x):
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
			
def tied_rank(x):
	"""
	refer https://statistics.laerd.com/statistical-guides/spearmans-rank-order-correlation-statistical-guide.php for details about ranking tied data

	inputs:
		x: list of num or 1D np.ndarray
	return:
		tied_rank of x: list of num
	"""
	total_num = len(x)
	x_sorted = [[e, r] for e, r in zip(sorted(x, reverse=True), range(total_num))]
	
	start_e = x_sorted[0][0]
	start_rank = 0
	for e, r in x_sorted:
		if e == start_e:
			pass

		else:
			mean_rank = np.mean(np.asarray(range(start_rank, r)))
			for idx in range(start_rank, r):
				x_sorted[idx][1] = mean_rank
			start_rank = r
			start_e = e
		
	mean_rank = np.mean(np.asarray(range(start_rank, total_num)))
	
	for idx in range(start_rank, total_num):
		x_sorted[idx][1] = mean_rank

	x_sorted_dict = dict(x_r for x_r in x_sorted)

	return [x_sorted_dict[e]+1 for e in x]



if __name__ == '__main__':
	x = np.arange(10)
	english = [56, 75, 45, 71, 61, 64, 58, 80, 76, 61]
	math = [66, 70, 40, 60, 65, 56, 59, 77, 67, 63]

	eng_rank = tied_rank(english)
	math_rank = tied_rank(math)
	print eng_rank
	print math_rank