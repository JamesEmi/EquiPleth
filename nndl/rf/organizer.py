import numpy as np 
from datetime import datetime as dt
from backports.datetime_fromisoformat import MonkeyPatch
MonkeyPatch.patch_fromisoformat()

MAX_PACKET_SIZE = 4096
BYTES_IN_PACKET = 1456

np.set_printoptions(threshold=np.inf,linewidth=325)

class Organizer:

	def __init__(self, all_data, num_chirp_loops, num_rx, num_tx, num_samples, verbose=False):
		self.data = all_data[0]
		self.packet_num = all_data[1]
		self.byte_count = all_data[2]

		self.num_packets = len(self.byte_count) #easily can be frame count also.
		self.num_chirps = num_chirp_loops*num_tx
		self.num_rx = num_rx
		self.num_samples = num_samples

		self.BYTES_IN_FRAME = self.num_chirps * self.num_rx * self.num_samples * 2 * 2
		self.BYTES_IN_FRAME_CLIPPED = (self.BYTES_IN_FRAME // BYTES_IN_PACKET) * BYTES_IN_PACKET
		self.UINT16_IN_FRAME = self.BYTES_IN_FRAME // 2
		self.NUM_PACKETS_PER_FRAME = self.BYTES_IN_FRAME // BYTES_IN_PACKET

		self.start_time = all_data[3]
		self.end_time = all_data[4]
		self.verbose = verbose

	def iq(self, raw_frame):
		"""Reorganizes raw ADC data into a full frame

		Args:
			raw_frame (ndarray): Data to format
			num_chirps: Number of chirps included in the frame
			num_rx: Number of receivers used in the frame
			num_samples: Number of ADC samples included in each chirp

		Returns:
			ndarray: Reformatted frame of raw data of shape (num_chirps, num_rx, num_samples)

		"""
		ret = np.zeros(len(raw_frame) // 2, dtype=np.csingle) #each pair of I/Q components will be combined into a single complex number, the resulting ret array needs only half as many elements as raw_frame

		# Separate IQ data
		ret[0::2] = raw_frame[0::4] + 1j * raw_frame[2::4]
		ret[1::2] = raw_frame[1::4] + 1j * raw_frame[3::4]
		return ret.reshape((self.num_chirps, self.num_rx, self.num_samples))

	def get_frames(self, start_chunk, end_chunk, bc):

		# if first packet received is not the first byte transmitted
		if bc[start_chunk] == 0:
			bytes_left_in_curr_frame = 0
			start = start_chunk*(BYTES_IN_PACKET // 2)
		else:
			frames_so_far = bc[start_chunk] // self.BYTES_IN_FRAME
			bytes_so_far = frames_so_far * self.BYTES_IN_FRAME
			# bytes_left_in_curr_frame = bc[start_chunk] - bytes_so_far
			bytes_left_in_curr_frame = (frames_so_far+1)*self.BYTES_IN_FRAME - bc[start_chunk]
			start = (bytes_left_in_curr_frame // 2) + start_chunk*(BYTES_IN_PACKET // 2)

		# if self.verbose: print(start_chunk, start)

		# find num of frames
		total_bytes = bc[end_chunk] - (bc[start_chunk] + bytes_left_in_curr_frame)
		num_frames = total_bytes // (self.BYTES_IN_FRAME)

		# if self.verbose: print(bc[end_chunk])
		# if self.verbose: print(num_frames, start_chunk, end_chunk, self.BYTES_IN_FRAME)
		frames = np.zeros((num_frames, self.UINT16_IN_FRAME), dtype=np.int16)
		ret_frames = np.zeros((num_frames, self.num_chirps, self.num_rx, self.num_samples), dtype=complex)		

		# compress all received data into one byte stream
		all_uint16 = np.array(self.data).reshape(-1)

		# only choose uint16 starting from a new frame
		all_uint16 = all_uint16[start:]

		# organizing into frames
		for i in range(num_frames):
			frame_start_idx = i*self.UINT16_IN_FRAME
			frame_end_idx = (i+1)*self.UINT16_IN_FRAME
			frame = all_uint16[frame_start_idx:frame_end_idx]
			frames[i][:len(frame)] = frame.astype(np.int16)
			ret_frames[i] = self.iq(frames[i])	

		return ret_frames

	
	def organize(self):

		radar_unix_start_time = dt.timestamp(dt.fromisoformat(self.start_time[:-1]))*1e6
		radar_unix_end_time = dt.timestamp(dt.fromisoformat(self.end_time[:-1]))*1e6

		if self.verbose: print('Start time: ', self.start_time)
		if self.verbose: print('End time: ', self.end_time)

		self.byte_count = np.array(self.byte_count)
		self.data = np.array(self.data)
		self.packet_num = np.array(self.packet_num)

		# Reordering packets
		# sorted_idx = np.argsort(self.packet_num)
		# if self.verbose: print(sorted_idx.dtype)
		# if self.verbose: print(len(self.packet_num), len(self.byte_count), len(self.data), sorted_idx.shape)
		# self.packet_num = self.packet_num[sorted_idx]
		# self.data = self.data[sorted_idx]
		# self.byte_count = self.byte_count[sorted_idx]

		# self.packet_num = self.packet_num.tolist()
		# self.byte_count = self.byte_count.tolist()
		# self.data = self.data.tolist()



		bc = np.array(self.byte_count)

		packets_ooo = np.where(np.array(self.packet_num[1:])-np.array(self.packet_num[0:-1]) != 1)[0]
		is_not_monotonic = np.where(np.array(self.packet_num[1:])-np.array(self.packet_num[0:-1]) < 0)[0]

		if self.verbose: print('Non monotonic packets: ', is_not_monotonic)

		if len(packets_ooo) == 0:
			if self.verbose: print('packets in order')
			start_chunk = 0
			ret_frames = self.get_frames(start_chunk, -1, bc)

		elif len(packets_ooo) == 1:
			if self.verbose: print('1 packet not in order')
			start_chunk = packets_ooo[0]+1
			ret_frames = self.get_frames(start_chunk, -1, bc)
			# start_chunk = 0

		else:
			if self.verbose: print('Packet num not in order')
			packets_ooo = np.append(packets_ooo, len(self.packet_num)-1)

			# if self.verbose: print('Packets ooo', packets_ooo)

			# if self.verbose: print(self.NUM_PACKETS_PER_FRAME)
			# diff = [44]
			# for i in range(len(packets_ooo)-1):
			# 	# if self.verbose: print(i, len(packets_ooo))
			# 	diff.append(self.packet_num[packets_ooo[i+1]]-self.packet_num[packets_ooo[i]+1])
			
			# if self.verbose: print('Packets received before atleast 1 loss ', diff)
			# if self.verbose: print('Total packets received ', np.sum(np.array(diff)))

			diff = []
			for i in range(len(packets_ooo)-1):
				diff.append(self.packet_num[packets_ooo[i]+1]-self.packet_num[packets_ooo[i]])
			
			# if self.verbose: print('Packets lost before atleast 1 reception ', diff)
			packets_lost = np.sum(np.array(diff))

			packets_expected = self.packet_num[-1]-self.packet_num[0]+1
			if self.verbose: print('Total packets lost ', packets_lost)
			if self.verbose: print('Total packets expected ', packets_expected)
			if self.verbose: print('Fraction lost ', packets_lost/packets_expected)

			new_packets_ooo = []
			start_new_packets_ooo = []
			end_new_packets_ooo = []
			for i in range(1, len(packets_ooo)):
				if (packets_ooo[i] - packets_ooo[i-1]) > self.NUM_PACKETS_PER_FRAME*2:
					new_packets_ooo.append(packets_ooo[i-1])
					start_new_packets_ooo.append(packets_ooo[i-1])
					end_new_packets_ooo.append(packets_ooo[i])

			new_packets_ooo = np.append(new_packets_ooo, -1)

			# if self.verbose: print('New packets ooo', new_packets_ooo)
			# if self.verbose: print('Start new packets ooo', start_new_packets_ooo)
			# if self.verbose: print('End new packets ooo', end_new_packets_ooo)
			# exit()

			for i in range(len(start_new_packets_ooo)):
			# for i in range(len(new_packets_ooo)-1):
			# for i in [len(new_packets_ooo)-2]:
				# start_chunk = new_packets_ooo[i]+1
				# end_chunk = new_packets_ooo[i+1]

				start_chunk = start_new_packets_ooo[i]+1
				end_chunk = end_new_packets_ooo[i]

				# if self.verbose: print(self.packet_num[start_chunk],self.packet_num[start_chunk-1])
				# if self.verbose: print(self.byte_count[start_chunk],self.byte_count[start_chunk-1])

				curr_frames = self.get_frames(start_chunk, end_chunk, bc)

				if i == 0:
					ret_frames = curr_frames
				else:
					ret_frames = np.concatenate((ret_frames, curr_frames), axis=0)


		return ret_frames

		# Old approach


		# frame_start_idx = np.where((bc % self.BYTES_IN_FRAME_CLIPPED == 0) & (bc != 0))[0]
		# num_frames = len(frame_start_idx)-1

		# frames = np.zeros((num_frames, self.UINT16_IN_FRAME), dtype=np.int16)
		# ret_frames = np.zeros((num_frames, self.num_chirps, self.num_rx, self.num_samples), dtype=complex)

		# for i in range(num_frames):
		# 	d = np.array(self.data[frame_start_idx[i]:frame_start_idx[i+1]])
		# 	frame = d.reshape(-1)
		# 	frames[i][:len(frame)] = frame.astype(np.int16)
		# 	ret_frames[i] = self.iq(frames[i])

		# return ret_frames
	
class Organizer_Chiron:

	def __init__(self, all_data, num_chirp_loops, num_rx, num_tx, num_samples, verbose=False, nsecs=30):
		self.nsecs = nsecs
		self.slice_size = int(len(all_data[1])/60 * self.nsecs) #when input data is worth 60s of data
		self.data = all_data[0][:self.slice_size]
		self.packet_num = all_data[1][:self.slice_size] #using frame_counter as a proxy
		# self.data = all_data[0]
		# self.packet_num = all_data[1] #using frame_counter as a proxy
		# self.byte_count = all_data[2]
		byte_count_step = 1500 
		self.byte_count = np.arange(257608624, 257608624 + len(self.packet_num) * byte_count_step, byte_count_step) #simulated array based on data from 91_1 pkl file in EP data

		self.num_packets = len(self.byte_count) #easily can be frame count also.
		self.num_chirps = num_chirp_loops*num_tx
		self.num_rx = num_rx
		self.num_samples = num_samples #what is this value supposed to be in Chiron case?

		self.BYTES_IN_FRAME = self.num_chirps * self.num_rx * self.num_samples * 2 * 2
		self.BYTES_IN_FRAME_CLIPPED = (self.BYTES_IN_FRAME // BYTES_IN_PACKET) * BYTES_IN_PACKET
		self.UINT16_IN_FRAME = self.BYTES_IN_FRAME // 2
		self.NUM_PACKETS_PER_FRAME = self.BYTES_IN_FRAME // BYTES_IN_PACKET

		# self.start_time = all_data[3]
		# self.end_time = all_data[4]
		self.verbose = verbose

	def iq(self, raw_frame):
		"""Reorganizes raw ADC data into a full frame

		Args:
			raw_frame (ndarray): Data to format
			num_chirps: Number of chirps included in the frame
			num_rx: Number of receivers used in the frame
			num_samples: Number of ADC samples included in each chirp

		Returns:
			ndarray: Reformatted frame of raw data of shape (num_chirps, num_rx, num_samples)

		"""
		ret = np.zeros(len(raw_frame) // 2, dtype=np.csingle) #each pair of I/Q components will be combined into a single complex number, the resulting ret array needs only half as many elements as raw_frame

		# Separate IQ data
		ret[0::2] = raw_frame[0::4] + 1j * raw_frame[2::4]
		ret[1::2] = raw_frame[1::4] + 1j * raw_frame[3::4]
		return ret.reshape((self.num_chirps, self.num_rx, self.num_samples))

	def get_frames(self, start_chunk, end_chunk, bc):

		# if first packet received is not the first byte transmitted
		if bc[start_chunk] == 0:
			bytes_left_in_curr_frame = 0
			start = start_chunk*(BYTES_IN_PACKET // 2)
		else:
			frames_so_far = bc[start_chunk] // self.BYTES_IN_FRAME
			bytes_so_far = frames_so_far * self.BYTES_IN_FRAME
			# bytes_left_in_curr_frame = bc[start_chunk] - bytes_so_far
			bytes_left_in_curr_frame = (frames_so_far+1)*self.BYTES_IN_FRAME - bc[start_chunk]
			start = (bytes_left_in_curr_frame // 2) + start_chunk*(BYTES_IN_PACKET // 2)

		# if self.verbose: print(start_chunk, start)

		# find num of frames
		total_bytes = bc[end_chunk] - (bc[start_chunk] + bytes_left_in_curr_frame)
		num_frames = total_bytes // (self.BYTES_IN_FRAME)

		# if self.verbose: print(bc[end_chunk])
		# if self.verbose: print(num_frames, start_chunk, end_chunk, self.BYTES_IN_FRAME)
		frames = np.zeros((num_frames, self.UINT16_IN_FRAME), dtype=np.int16)
		ret_frames = np.zeros((num_frames, self.num_chirps, self.num_rx, self.num_samples), dtype=complex)		

		# compress all received data into one byte stream
		all_uint16 = np.array(self.data).reshape(-1)

		# only choose uint16 starting from a new frame
		all_uint16 = all_uint16[start:]

		# organizing into frames
		for i in range(num_frames):
			frame_start_idx = i*self.UINT16_IN_FRAME
			frame_end_idx = (i+1)*self.UINT16_IN_FRAME
			frame = all_uint16[frame_start_idx:frame_end_idx]
			frames[i][:len(frame)] = frame.astype(np.int16)
			ret_frames[i] = self.iq(frames[i])	

		return ret_frames

	def get_frames_ch(self, start_chunk, end_chunk, byte_count, adc_samples):
			"""
			Extracts and reshapes frames from the data.

			:param start_chunk: Starting index for the frames.
			:param end_chunk: Ending index for the frames.
			:param byte_count: Byte count for each packet.
			:param adc_samples: Number of ADC samples.
			:return: Reshaped data into (n, 1, 1, 2*adc_samples).
			"""
			# Simulate frame extraction from data
			if end_chunk == -1:
				end_chunk = len(self.data)

			# Extract the required range of data
			raw_data_chunk = np.array(self.data[start_chunk:end_chunk])

			# Reshape data into desired format
			n = raw_data_chunk.shape[0] // (2 * adc_samples)
			reshaped_data = raw_data_chunk.reshape((n, 1, 1, 2 * adc_samples))

			return reshaped_data

	def organize(self):

		# radar_unix_start_time = dt.timestamp(dt.fromisoformat(self.start_time[:-1]))*1e6
		# radar_unix_end_time = dt.timestamp(dt.fromisoformat(self.end_time[:-1]))*1e6

		if self.verbose: print('Start time: ', self.start_time)
		if self.verbose: print('End time: ', self.end_time)

		self.byte_count = np.array(self.byte_count)
		self.data = np.array(self.data)
		self.packet_num = np.array(self.packet_num)

		# Reordering packets
		# sorted_idx = np.argsort(self.packet_num)
		# if self.verbose: print(sorted_idx.dtype)
		# if self.verbose: print(len(self.packet_num), len(self.byte_count), len(self.data), sorted_idx.shape)
		# self.packet_num = self.packet_num[sorted_idx]
		# self.data = self.data[sorted_idx]
		# self.byte_count = self.byte_count[sorted_idx]

		# self.packet_num = self.packet_num.tolist()
		# self.byte_count = self.byte_count.tolist()
		# self.data = self.data.tolist()



		bc = np.array(self.byte_count)

		packets_ooo = np.where(np.array(self.packet_num[1:])-np.array(self.packet_num[0:-1]) != 1)[0]
		is_not_monotonic = np.where(np.array(self.packet_num[1:])-np.array(self.packet_num[0:-1]) < 0)[0]

		if self.verbose: print('Non monotonic packets: ', is_not_monotonic)

		if len(packets_ooo) == 0:
			if self.verbose: print('packets in order')
			start_chunk = 0
			ret_frames = self.get_frames(start_chunk, -1, bc)

		elif len(packets_ooo) == 1:
			if self.verbose: print('1 packet not in order')
			start_chunk = packets_ooo[0]+1
			ret_frames = self.get_frames(start_chunk, -1, bc)
			# start_chunk = 0

		else:
			if self.verbose: print('Packet num not in order')
			packets_ooo = np.append(packets_ooo, len(self.packet_num)-1)

			# if self.verbose: print('Packets ooo', packets_ooo)

			# if self.verbose: print(self.NUM_PACKETS_PER_FRAME)
			# diff = [44]
			# for i in range(len(packets_ooo)-1):
			# 	# if self.verbose: print(i, len(packets_ooo))
			# 	diff.append(self.packet_num[packets_ooo[i+1]]-self.packet_num[packets_ooo[i]+1])
			
			# if self.verbose: print('Packets received before atleast 1 loss ', diff)
			# if self.verbose: print('Total packets received ', np.sum(np.array(diff)))

			diff = []
			for i in range(len(packets_ooo)-1):
				diff.append(self.packet_num[packets_ooo[i]+1]-self.packet_num[packets_ooo[i]])
			
			# if self.verbose: print('Packets lost before atleast 1 reception ', diff)
			packets_lost = np.sum(np.array(diff))

			packets_expected = self.packet_num[-1]-self.packet_num[0]+1
			if self.verbose: print('Total packets lost ', packets_lost)
			if self.verbose: print('Total packets expected ', packets_expected)
			if self.verbose: print('Fraction lost ', packets_lost/packets_expected)

			new_packets_ooo = []
			start_new_packets_ooo = []
			end_new_packets_ooo = []
			for i in range(1, len(packets_ooo)):
				if (packets_ooo[i] - packets_ooo[i-1]) > self.NUM_PACKETS_PER_FRAME*2:
					new_packets_ooo.append(packets_ooo[i-1])
					start_new_packets_ooo.append(packets_ooo[i-1])
					end_new_packets_ooo.append(packets_ooo[i])

			new_packets_ooo = np.append(new_packets_ooo, -1)

			# if self.verbose: print('New packets ooo', new_packets_ooo)
			# if self.verbose: print('Start new packets ooo', start_new_packets_ooo)
			# if self.verbose: print('End new packets ooo', end_new_packets_ooo)
			# exit()

			for i in range(len(start_new_packets_ooo)):
			# for i in range(len(new_packets_ooo)-1):
			# for i in [len(new_packets_ooo)-2]:
				# start_chunk = new_packets_ooo[i]+1
				# end_chunk = new_packets_ooo[i+1]

				start_chunk = start_new_packets_ooo[i]+1
				end_chunk = end_new_packets_ooo[i]

				# if self.verbose: print(self.packet_num[start_chunk],self.packet_num[start_chunk-1])
				# if self.verbose: print(self.byte_count[start_chunk],self.byte_count[start_chunk-1])

				curr_frames = self.get_frames(start_chunk, end_chunk, bc)

				if i == 0:
					ret_frames = curr_frames
				else:
					ret_frames = np.concatenate((ret_frames, curr_frames), axis=0)


		return ret_frames

		# Old approach


		# frame_start_idx = np.where((bc % self.BYTES_IN_FRAME_CLIPPED == 0) & (bc != 0))[0]
		# num_frames = len(frame_start_idx)-1

		# frames = np.zeros((num_frames, self.UINT16_IN_FRAME), dtype=np.int16)
		# ret_frames = np.zeros((num_frames, self.num_chirps, self.num_rx, self.num_samples), dtype=complex)

		# for i in range(num_frames):
		# 	d = np.array(self.data[frame_start_idx[i]:frame_start_idx[i+1]])
		# 	frame = d.reshape(-1)
		# 	frames[i][:len(frame)] = frame.astype(np.int16)
		# 	ret_frames[i] = self.iq(frames[i])

		# return ret_frames

	def organize_chiron(self, adc_samples):
		"""
		Organizes Chiron data into a shape suitable for further processing.

		:param adc_samples: The number of ADC samples used in the radar data.
		:return: The organized data with shape (n, 1, 1, 2*adc_samples).
		"""
		# Simulate start and end time conversion to Unix timestamp (microseconds)
		if self.start_time and self.end_time:
			radar_unix_start_time = dt.timestamp(dt.fromisoformat(self.start_time[:-1])) * 1e6
			radar_unix_end_time = dt.timestamp(dt.fromisoformat(self.end_time[:-1])) * 1e6

			if self.verbose:
				print('Start time: ', self.start_time)
				print('End time: ', self.end_time)

		# self.packet_num = np.arange(len(self.data[1]))  # Assuming packet numbers are sequentially ordered
		
		self.data = np.array(self.data)
		self.packet_num = np.array(self.packet_num)
		self.byte_count = np.array([len(d) for d in self.data])  # Example byte count calculation
		# self.byte_count = np.array(self.byte_count)

		# Initialize variables for organizing the data
		bc = np.array(self.byte_count)
		packets_ooo = np.where(np.diff(self.packet_num) != 1)[0]
		is_not_monotonic = np.where(np.diff(self.packet_num) < 0)[0]

		if self.verbose:
			print('Non monotonic packets: ', is_not_monotonic)

		# Handling different packet order scenarios
		if len(packets_ooo) == 0:
			if self.verbose:
				print('Packets in order')
			start_chunk = 0
			ret_frames = self.get_frames(start_chunk, -1, bc, adc_samples)

		elif len(packets_ooo) == 1:
			if self.verbose:
				print('1 packet not in order')
			start_chunk = packets_ooo[0] + 1
			ret_frames = self.get_frames(start_chunk, -1, bc, adc_samples)

		else:
			if self.verbose:
				print('Packet num not in order')
			packets_ooo = np.append(packets_ooo, len(self.packet_num) - 1)

			diff = np.diff(packets_ooo)
			packets_lost = np.sum(diff)

			packets_expected = self.packet_num[-1] - self.packet_num[0] + 1
			if self.verbose:
				print('Total packets lost: ', packets_lost)
				print('Total packets expected: ', packets_expected)
				print('Fraction lost: ', packets_lost / packets_expected)

			new_packets_ooo = []
			start_new_packets_ooo = []
			end_new_packets_ooo = []
			for i in range(1, len(packets_ooo)):
				if (packets_ooo[i] - packets_ooo[i - 1]) > adc_samples * 2:
					new_packets_ooo.append(packets_ooo[i - 1])
					start_new_packets_ooo.append(packets_ooo[i - 1])
					end_new_packets_ooo.append(packets_ooo[i])

			new_packets_ooo = np.append(new_packets_ooo, -1)

			for i in range(len(start_new_packets_ooo)):
				start_chunk = start_new_packets_ooo[i] + 1
				end_chunk = end_new_packets_ooo[i]

				curr_frames = self.get_frames(start_chunk, end_chunk, bc, adc_samples)

				if i == 0:
					ret_frames = curr_frames
				else:
					ret_frames = np.concatenate((ret_frames, curr_frames), axis=0)

		return ret_frames
