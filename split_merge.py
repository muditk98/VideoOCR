import cv2


class VideoStreamer:
	def __init__(self, read_path, name):
		self.read_path = read_path
		self.reader = cv2.VideoCapture(read_path)
		self.fps = int(self.reader.get(cv2.CAP_PROP_FPS))
		self.width = int(self.reader.get(cv2.CAP_PROP_FRAME_WIDTH))
		self.height = int(self.reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
		self.writer = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*'H264'), self.fps, (self.width, self.height))
		self.total_frames = self.reader.get(cv2.CAP_PROP_FRAME_COUNT)
		self.count = 0

	def read_frame(self):
		success, frame = self.reader.read()
		if success:
			self.count += 1
			return frame
		else:
			return None

	def write_frame(self, frame):
		self.writer.write(frame)

	def finish(self):
		self.reader.release()
		self.writer.release()

	def progress(self):
		return self.count / self.total_frames


