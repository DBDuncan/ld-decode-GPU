#!/usr/bin/python3
import numpy as np
import scipy as sp
import scipy.signal as sps
import scipy.fftpack as fftpack 
import sys

import fdls as fdls
import matplotlib.pyplot as plt
import fft8 as fft8 

import getopt

π = np.pi
τ = np.pi * 2

#import ipdb

freq = (315.0 / 88.0) * 8.0
freq_hz = freq * 1000000.0

blocklen = (16 * 1024) + 512
hilbertlen = (16 * 1024)

wide_mode = 0

def dosplot(B, A):
	w, h = sps.freqz(B, A)

	fig = plt.figure()
	plt.title('Digital filter frequency response')

	ax1 = fig.add_subplot(111)

	db = 20 * np.log10(abs(h))

	for i in range(1, len(w)):
		if (db[i] >= -10) and (db[i - 1] < -10):
			print(">-10db crossing at ", w[i] * (freq/π) / 2.0) 
		if (db[i] >= -3) and (db[i - 1] < -3):
			print("-3db crossing at ", w[i] * (freq/π) / 2.0) 
		if (db[i] < -3) and (db[i - 1] >= -3):
			print("<-3db crossing at ", w[i] * (freq/π) / 2.0) 
		if (db[i] < -10) and (db[i - 1] >= -10):
			print("<-10db crossing at ", w[i] * (freq/π) / 2.0) 
		if (db[i] < -20) and (db[i - 1] >= -20):
			print("<-20db crossing at ", w[i] * (freq/π) / 2.0) 

	plt.plot(w * (freq/π) / 2.0, 20 * np.log10(abs(h)), 'b')
	plt.ylabel('Amplitude [dB]', color='b')
	plt.xlabel('Frequency [rad/sample]')

	plt.show()

def doplot(B, A):
	w, h = sps.freqz(B, A)

	fig = plt.figure()
	plt.title('Digital filter frequency response')
	
	db = 20 * np.log10(abs(h))
	for i in range(1, len(w)):
		if (db[i] >= -10) and (db[i - 1] < -10):
			print(">-10db crossing at ", w[i] * (freq/π) / 2.0) 
		if (db[i] >= -3) and (db[i - 1] < -3):
			print(">-3db crossing at ", w[i] * (freq/π) / 2.0) 
		if (db[i] < -3) and (db[i - 1] >= -3):
			print("<-3db crossing at ", w[i] * (freq/π) / 2.0) 

	ax1 = fig.add_subplot(111)
	
	plt.plot(w * (freq/π) / 2.0, 20 * np.log10(abs(h)), 'b')
	plt.ylabel('Amplitude [dB]', color='b')
	plt.xlabel('Frequency [rad/sample]')

	ax2 = ax1.twinx()
	angles = np.unwrap(np.angle(h))
	plt.plot(w * (freq/π) / 2.0, angles, 'g')
	plt.ylabel('Angle (radians)', color='g')
	
	plt.grid()
	plt.axis('tight')
	plt.show()

def doplot2(B, A, B2, A2):
	w, h = sps.freqz(B, A)
	w2, h2 = sps.freqz(B2, A2)

#	h.real /= C
#	h2.real /= C2

	begin = 0
	end = len(w)
#	end = int(len(w) * (12 / freq))

#	chop = len(w) / 20
	chop = 0
	w = w[begin:end]
	w2 = w2[begin:end]
	h = h[begin:end]
	h2 = h2[begin:end]

	v = np.empty(len(w))
	
#	print len(w)

	hm = np.absolute(h)
	hm2 = np.absolute(h2)

	v0 = hm[0] / hm2[0]
	for i in range(0, len(w)):
#		print i, freq / 2 * (w[i] / π), hm[i], hm2[i], hm[i] / hm2[i], (hm[i] / hm2[i]) / v0
		v[i] = (hm[i] / hm2[i]) / v0

	fig = plt.figure()
	plt.title('Digital filter frequency response')

	ax1 = fig.add_subplot(111)

	v  = 20 * np.log10(v )

#	plt.plot(w * (freq/π) / 2.0, v)
#	plt.show()
#	exit()

	plt.plot(w * (freq/π) / 2.0, 20 * np.log10(abs(h)), 'r')
	plt.plot(w * (freq/π) / 2.0, 20 * np.log10(abs(h2)), 'b')
	plt.ylabel('Amplitude [dB]', color='b')
	plt.xlabel('Frequency [rad/sample]')
	
	ax2 = ax1.twinx()
	angles = np.unwrap(np.angle(h))
	angles2 = np.unwrap(np.angle(h2))
	plt.plot(w * (freq/π) / 2.0, angles, 'g')
	plt.plot(w * (freq/π) / 2.0, angles2, 'y')
	plt.ylabel('Angle (radians)', color='g')

	plt.grid()
	plt.axis('tight')
	plt.show()

ffreq = freq/2.0

Bboost, Aboost = sps.butter(1, (2.0/(freq/2)), 'high')

lowpass_filter_b, lowpass_filter_a = sps.butter(8, (5.5/(freq/2)), 'low')

#dosplot(lowpass_filter_b, lowpass_filter_a)
#exit()

#lowpass_filter_b = [1.0]
#lowpass_filter_a = [1.0]

# demphasis coefficients.  haven't figured out how to compute them outside octave yet.

# [b, a] = bilinear(-3*(10^-8), -1*(10^-8), 1/3, freq)
f_deemp_b = [3.778720395899611e-01, -2.442559208200777e-01]
f_deemp_a = [1.000000000000000e+00, -8.663838812301168e-01]
deemp_corr = 1.0

# t1 = 0.83; t2 = 3.0; [b, a] = bilinear(-t2*(10^-8), -t1*(10^-8), t1/t2, freq);
# printf("f_deemp_b = ["); printf("%.15e, ", b); printf("]\nf_deemp_a = ["); printf("%.15e, ", a); printf("]\n")
#f_deemp_b = [3.172367682445019e-01, -2.050613721767547e-01, ]
#f_deemp_a = [1.000000000000000e+00, -8.878246039322528e-01, ]

# t1 = 0.9; t2 = 3.0; [b, a] = bilinear(-t2*(10^-8), -t1*(10^-8), t1/t2, freq);
# printf("f_deemp_b = ["); printf("%.15e, ", b); printf("]\nf_deemp_a = ["); printf("%.15e, ", a); printf("]\n")
#f_deemp_b = [3.423721575744635e-01, -2.213088502188534e-01, ]
#f_deemp_a = [1.000000000000000e+00, -8.789366926443899e-01, ]

# t1 = .875
#f_deemp_b = [3.334224479793254e-01, -2.155237713318184e-01, ]
#f_deemp_a = [1.000000000000000e+00, -8.821013233524929e-01, ]

# t1 = .833
#f_deemp_b = [3.183188754563553e-01, -2.057608446588788e-01, ]
#f_deemp_a = [1.000000000000000e+00, -8.874419692025236e-01, ]

# t1 = .8
#f_deemp_b = [3.063915161937518e-01, -1.980510174835196e-01, ]
#f_deemp_a = [1.000000000000000e+00, -8.916595012897678e-01, ]

# t1 = 0.9, t2 = 4.0 - PAL
f_deemp_b = [2.250000998693053e-01, -2.249998421420658e-01, ]
f_deemp_a = [1.000000000000000e+00, -9.999997422727605e-01, ]

#f_deemp_b = [2.500001073863483e-01, -2.499998210227529e-01, ]
#f_deemp_a = [1.000000000000000e+00, -9.999997136364046e-01, ]
f_deemp_b = [3.001060445387063e-01, -1.664899257688229e-01, ]
f_deemp_a = [1.000000000000000e+00, -8.663838812301168e-01, ]

f_deemp_b = [2.719120316002989e-01, -1.508487242446888e-01, ]
f_deemp_a = [1.000000000000000e+00, -8.789366926443899e-01, ]

# t1 = 0.7, t2 = 4.0 - PAL
f_deemp_b = [2.433361994840929e-01, -1.349957007738607e-01, ]
f_deemp_a = [1.000000000000000e+00, -8.916595012897678e-01, ]
f_deemp_b = [2.143707390975003e-01, -1.189265231035602e-01, ]
f_deemp_a = [1.000000000000000e+00, -9.045557840060601e-01, ]

f_deemp_b = [1.958542245307747e-01, -1.004100085368346e-01, ]
f_deemp_a = [1.000000000000000e+00, -9.045557840060601e-01, ]

f_deemp_b = [2.381776863975760e-01, -1.427334704036360e-01, ]
f_deemp_a = [1.000000000000000e+00, -9.045557840060601e-01, ]

# freq = 4000000*315/88
# t1 = 0.7; t2 = 4.0; [b, a] = bilinear(-t2*(10^-8), -t1*(10^-8), t1/t2, freq); printf("f_deemp_b = ["); printf("%.1
# 5e, ", b); printf("]\nf_deemp_a = ["); printf("%.15e, ", a); printf("]\n")
#f_deemp_b = [2.143707390975003e-01, -1.189265231035602e-01, ]
#f_deemp_a = [1.000000000000000e+00, -9.045557840060601e-01, ]

# t1 0.8 t2 4.0
#f_deemp_b = [2.433361994840929e-01, -1.349957007738607e-01, ]
#f_deemp_a = [1.000000000000000e+00, -8.916595012897678e-01, ]

# t1 0.75
f_deemp_b = [2.289026691830682e-01, -1.269884065785926e-01, ]
f_deemp_a = [1.000000000000000e+00, -8.980857373955243e-01, ]

# audio filters
Baudiorf = sps.firwin(65, 3.5 / (freq / 2), window='hamming', pass_zero=True)

afreq = freq / 4

left_audfreq = 2.301136
right_audfreq = 2.812499

hfreq = freq / 8.0

N, Wn = sps.buttord([(left_audfreq-.10) / hfreq, (left_audfreq+.10) / hfreq], [(left_audfreq-.15) / hfreq, (left_audfreq+.15)/hfreq], 1, 15)
#print(N,Wn)
leftbp_filter_b, leftbp_filter_a = sps.butter(N, Wn, btype='bandpass')

N, Wn = sps.buttord([(right_audfreq-.10) / hfreq, (right_audfreq+.10) / hfreq], [(right_audfreq-.15) / hfreq, (right_audfreq+.15)/hfreq], 1, 15)
#print(N,Wn)
#N, Wn = sps.buttord([(right_audfreq-.10) / hfreq, (right_audfreq+.10) / hfreq], [(right_audfreq-.18) / hfreq, (right_audfreq+.18)/hfreq], 1, 15)
#print(N,Wn)
rightbp_filter_b, rightbp_filter_a = sps.butter(N, Wn, btype='bandpass')

#doplot(leftbp_filter_b, leftbp_filter_a)
#doplot2(leftbp_filter, [1.0], rightbp_filter, [1.0]);
#doplot2(leftbp_filter_b, leftbp_filter_a, rightbp_filter_b, rightbp_filter_a);

N, Wn = sps.buttord(0.016 / hfreq, 0.024 / hfreq, 1, 8) 
audiolp_filter_b, audiolp_filter_a = sps.butter(N, Wn)

N, Wn = sps.buttord(3.1 / (freq / 2.0), 3.5 / (freq / 2.0), 1, 16) 
#N = 12
#Wn = 3.3 / (freq / 2.0)
#N = 8
#Wn = 3.4 / (freq / 2.0)
audiorf_filter_b, audiorf_filter_a = sps.butter(N, Wn)

# from http://tlfabian.blogspot.com/2013/01/implementing-hilbert-90-degree-shift.html
hilbert_filter = np.fft.fftshift(
    np.fft.ifft([0]+[1]*20+[0]*20)
)

def fm_decode(in_filt, freq_hz, hlen = hilbertlen):
	hilbert = sps.hilbert(in_filt[0:hlen])
#	hilbert = sps.lfilter(hilbert_filter, 1.0, in_filt)

	# the hilbert transform has errors at the edges.  but it doesn't seem to matter much IRL
	chop = 256 
	hilbert = hilbert[chop:len(hilbert)-chop]

	tangles = np.angle(hilbert) 

#	dangles = np.diff(tangles[128:])
	dangles = np.diff(tangles)

	# make sure unwapping goes the right way
	if (dangles[0] < -π):
		dangles[0] += τ
	
	tdangles2 = np.unwrap(dangles) 
	
	output = (tdangles2 * (freq_hz / τ))

	errcount = 1
	while errcount > 0:
		errcount = 0

		# particularly bad bits can cause phase inversions.  detect and fix when needed - the loops are slow in python.
		if (output[np.argmax(output)] > freq_hz):
			errcount = 1
			for i in range(0, len(output)):
				if output[i] > freq_hz:
					output[i] = output[i] - freq_hz
	
		if (output[np.argmin(output)] < 0):
			errcount = 1
			for i in range(0, len(output)):
				if output[i] < 0:
					output[i] = output[i] + freq_hz

	return output

minire = -95
maxire = 145

hz_ire_scale = (8000000 - 7100000) / 100
minn = 7100000 + (hz_ire_scale * minire)

out_scale = 65534.0 / (maxire - minire)
	
#doplot(Bcutl, Acutl)

# octave:104> t1 = 100; t2 = 55; [b, a] = bilinear(-t2*(10^-8), -t1*(10^-8), t1/t2, freq); freqz(b, a)
# octave:105> printf("f_emp_b = ["); printf("%.15e, ", b); printf("]\nf_emp_a = ["); printf("%.15e, ", a); printf("]\n")
f_emp_b = [1.293279022403258e+00, -1.018329938900196e-02, ]
f_emp_a = [1.000000000000000e+00, 2.830957230142566e-01, ]

Inner = 0

#lowpass_filter_b = [1.0]
#lowpass_filter_a = [1.0]


def process_video(data):
	# perform general bandpass filtering

	in_filt = sps.lfilter(Bboost, Aboost, data)
#	in_filt2 = sps.lfilter(Bcutl, Acutl, in_filt1)
#	in_filt3 = sps.lfilter(Bcutr, Acutr, in_filt2)

#	if Inner == 0:
#		in_filt = in_filt3
#	else:
#		in_filt = sps.lfilter(f_emp_b, f_emp_a, in_filt3)


#	fft8.plotfft(in_filt)
#	#fft8.plotfft(data)
#	plt.plot(in_filt)
#	plt.show()

	output = fm_decode(in_filt, freq_hz)

	# save the original fm decoding and align to filters
	output_prefilt = output[(len(f_deemp_b) * 24) + len(f_deemp_b) + len(lowpass_filter_b):]

	# clip impossible values (i.e. rot)
#	output = np.clip(output, 7100000, 10800000) 

	output = sps.lfilter(lowpass_filter_b, lowpass_filter_a, output)

	doutput = (sps.lfilter(f_deemp_b, f_deemp_a, output)[len(f_deemp_b) * 32:len(output)]) / deemp_corr

#	doutput = (sps.lfilter(f_deemp_b, f_deemp_a, doutput)[64:len(doutput)]) / deemp_corr
	
	output_16 = np.empty(len(doutput), dtype=np.uint16)
	reduced = (doutput - minn) / hz_ire_scale
	output = np.clip(reduced * out_scale, 0, 65535) 
	
	np.copyto(output_16, output, 'unsafe')

	err = 1
	while False: # err > 0:
		err = 0
	
		am = np.argmax(output_prefilt)	
		if (output_prefilt[np.argmax(output_prefilt)] > 11200000):
			err = 1
			output_prefilt[am] = 11200000
			if (am < len(output_16)):
				output_16[am] = 0
		
		am = np.argmin(output_prefilt)	
		if (output_prefilt[np.argmax(output_prefilt)] < 3000000):
			err = 1
			output_prefilt[am] = 3000000
			if (am < len(output_16)):
				output_16[am] = 0

#	plt.plot(range(0, len(data)), data)
#	plt.plot(range(0, len(output_16)), output_16)
#	plt.show()
	return output_16

# graph for debug
#	output = (sps.lfilter(f_deemp_b, f_deemp_a, output)[128:len(output)]) / deemp_corr

#	print output

	plt.plot(range(0, len(output_16)), output_16)
#	plt.plot(range(0, len(doutput)), doutput)
#	plt.plot(range(0, len(output_prefilt)), output_prefilt)
	plt.show()
	exit()

left_audfreqm = left_audfreq * 1000000
right_audfreqm = right_audfreq * 1000000

test_mode = 0

def process_audio(indata):
	global test_mode

	if test_mode > 0:
		outputf = np.empty(32768 * 2, dtype = np.float32)
		for i in range(0, 32768):
			outputf[i * 2] = np.cos((i + test_mode) / (freq_hz / 4.0 / 10000)) 
			outputf[(i * 2) + 1] = np.cos((i + test_mode) / (freq_hz / 4.0 / 10000)) 

		outputf *= 50000
	
		test_mode += 32768 
		return outputf, 32768 

#	print(len(indata), len(audiorf_filter_b * 2), len(leftbp_filter_b) * 1)

	in_filt = sps.lfilter(audiorf_filter_b, audiorf_filter_a, indata)[len(audiorf_filter_b) * 2:]

	in_filt4 = np.empty(int(len(in_filt) / 4) + 1)

	for i in range(0, len(in_filt), 4):
		in_filt4[int(i / 4)] = in_filt[i]

	in_left = sps.lfilter(leftbp_filter_b, leftbp_filter_a, in_filt4)[len(leftbp_filter_b) * 1:] 
	in_right = sps.lfilter(rightbp_filter_b, rightbp_filter_a, in_filt4)[len(rightbp_filter_b) * 1:] 

#	if (len(in_left) % 2):
#		in_left = in_left[0:len(in_left - 1)]
#	if (len(in_right) % 2):
#		in_right = in_right[0:len(in_right - 1)]

#	print len(in_left)

	out_left = fm_decode(in_left, freq_hz / 4)
	out_right = fm_decode(in_right, freq_hz / 4)

	out_left = np.clip(out_left - left_audfreqm, -150000, 150000) 
	out_right = np.clip(out_right - right_audfreqm, -150000, 150000) 

	out_left = sps.lfilter(audiolp_filter_b, audiolp_filter_a, out_left)[800:]
	out_right = sps.lfilter(audiolp_filter_b, audiolp_filter_a, out_right)[800:] 

	outputf = np.empty((len(out_left) * 2.0 / 20.0) + 2, dtype = np.float32)

	tot = 0
	for i in range(0, len(out_left), 20):
		outputf[tot * 2] = out_left[i]
		outputf[(tot * 2) + 1] = out_right[i]
		tot = tot + 1

#	exit()
	
	return outputf[0:tot * 2], tot * 20 * 4 

	plt.plot(range(0, len(out_left)), out_left)
#	plt.plot(range(0, len(out_leftl)), out_leftl)
	plt.plot(range(0, len(out_right)), out_right + 150000)
#	plt.ylim([2000000,3000000])
	plt.show()
	exit()

def test():
	test = np.empty(blocklen, dtype=np.uint8)
#	test = np.empty(blocklen)

	global hilbert_filter

#	for hlen in range(3, 18):
	for vlevel in range(20, 100, 10):
#		hilbert_filter = np.fft.fftshift(
#		    np.fft.ifft([0]+[1]*hlen+[0]*hlen)
#		)

#		vlevel = 40
		vphase = 0
		alphase = 0
		arphase = 0

		for i in range(0, len(test)):
			if i > len(test) / 2:
				vfreq = 9300000
			else:
				vfreq = 8100000

			vphase += vfreq / freq_hz 
			alphase += 2300000 / freq_hz 
			arphase += 2800000 / freq_hz 
			test[i] = (np.sin(vphase * τ) * vlevel)
			test[i] += (np.sin(alphase * τ) * vlevel / 10.0)
			test[i] += (np.sin(arphase * τ) * vlevel / 10.0)
			test[i] += 128

		output = process_video(test)[7800:8500]	
		plt.plot(range(0, len(output)), output)

		output = output[400:700]	
		mean = np.mean(output)
		std = np.std(output)
		print(vlevel, mean, std, 20 * np.log10(mean / std)) 

	plt.show()
	exit()

def main():
	global lowpass_filter_b, lowpass_filter_a 
	global wide_mode, hz_ire_scale, minn
	global f_deemp_b, f_deemp_a

	global Bcutr, Acutr
	
	global Inner 

	global blocklen

#	test()

	outfile = sys.stdout.buffer
	audio_mode = 0 
	CAV = 0

	byte_start = 0
	byte_end = 0

	f_seconds = False 

	optlist, cut_argv = getopt.getopt(sys.argv[1:], "hLCaAwSs:")

	for o, a in optlist:
		if o == "-a":
			audio_mode = 1	
			blocklen = (64 * 1024) + 2048 
			hilbertlen = (16 * 1024)
		if o == "-L":
			Inner = 1
		if o == "-A":
			CAV = 1
			Inner = 1
		if o == "-h":
			# use full spec deemphasis filter - will result in overshoot, but higher freq resonse
			f_deemp_b = [3.778720395899611e-01, -2.442559208200777e-01]
			f_deemp_a = [1.000000000000000e+00, -8.663838812301168e-01]
		if o == "-C":
			Bcutr, Acutr = sps.butter(1, [2.68/(freq/2), 3.08/(freq/2)], btype='bandstop')
		if o == "-w":
#			lowpass_filter_b, lowpass_filter_a = sps.butter(9, (5.0/(freq/2)), 'low')
#			lowpass_filter_b, lowpass_filter_a = sps.butter(8, (4.8/(freq/2)), 'low')
			lowpass_filter_b, lowpass_filter_a = sps.butter(7, (4.7/(freq/2)), 'low')
			wide_mode = 1
			hz_ire_scale = (9360000 - 8100000) / 100
			minn = 8100000 + (hz_ire_scale * -60)
		if o == "-S":
			f_seconds = True
		if o == "-s":
			ia = int(a)
			if ia == 0:
				lowpass_filter_b, lowpass_filter_a = sps.butter(8, (5.2/(freq/2)), 'low')
			if ia == 1:	
				lowpass_filter_b, lowpass_filter_a = sps.butter(8, (5.5/(freq/2)), 'low')
			if ia == 2:	
				lowpass_filter_b, lowpass_filter_a = sps.butter(8, (5.8/(freq/2)), 'low')
			if ia == 3:	
				lowpass_filter_b, lowpass_filter_a = sps.butter(8, (6.1/(freq/2)), 'low')


#	dosplot(lowpass_filter_b, lowpass_filter_a)
#	exit()

	argc = len(cut_argv)
	if argc >= 1:
		infile = open(cut_argv[0], "rb")
	else:
		infile = sys.stdin

	byte_start = 0
	if (argc >= 2):
		byte_start = float(cut_argv[1])

	if (argc >= 3):
		byte_end = float(cut_argv[2])
		limit = 1
	else:
		limit = 0

	if f_seconds:
		byte_start *= freq_hz 
		byte_end *= freq_hz 
	else:
		# for backwards compat (for now)
		byte_end += byte_start

	byte_end -= byte_start

	byte_start = int(byte_start)
	byte_end = int(byte_end)

	if (byte_start > 0):	
		infile.seek(byte_start)
	
	if CAV and byte_start > 11454654400:
		CAV = 0
		Inner = 0 
	
	total = toread = blocklen 
	inbuf = infile.read(toread)
	indata = np.fromstring(inbuf, 'uint8', toread)
	
	total = 0
	total_prevread = 0
	total_read = 0

	while (len(inbuf) > 0):
		toread = blocklen - indata.size 

		if toread > 0:
			inbuf = infile.read(toread)
			indata = np.append(indata, np.fromstring(inbuf, 'uint8', len(inbuf)))

			if indata.size < blocklen:
				exit()

		if audio_mode:	
			output, osamp = process_audio(indata)
#			for i in range(0, osamp):
#				print i, output[i * 2], output[(i * 2) + 1]
			
			nread = osamp 

#			print len(output), nread

			outfile.write(output)
		else:
			output_16 = process_video(indata)
#			output_16.tofile(outfile)
			outfile.write(output_16)
			nread = len(output_16)
			
#			print(len(output_16), nread)

			total_pread = total_read 
			total_read += nread

			if CAV:
				if (total_read + byte_start) > 11454654400:
					CAV = 0
					Inner = 0

		indata = indata[nread:len(indata)]

		if limit == 1:
			byte_end -= toread 
			if (byte_end < 0):
				inbuf = ""

if __name__ == "__main__":
    main()


