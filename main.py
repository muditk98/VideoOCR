import underkill
import split_merge

streamer = split_merge.VideoStreamer('test_video.mp4', 'converted2.mp4')

while True:
	frame = streamer.read_frame()
	if frame is None:
		break
	streamer.write_frame(underkill.process(frame))
	print("-"*10 + str(streamer.progress()) + '-'*10)

streamer.finish()
