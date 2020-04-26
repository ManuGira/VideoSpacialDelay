import cv2 as cv
import numpy as np
import time

class VideoSpacialDelay:
    def __init__(self):
        self.frame_stack = None
        self.stack_length = 100
        self.stack_index = -1

    def process(self):
        H, W, N = self.frame_stack.shape
        dtype = self.frame_stack.dtype
        out = np.zeros(shape=(H, W), dtype=dtype)

        for n in range(N):
            h0 = int(round(H * n/N))
            h1 = int(round(H * (n+1)/N))
            nc = (self.stack_index + n) % self.stack_length
            out[h0:h1, :] = self.frame_stack[h0:h1, :, nc]
        return out

    def run(self):
        cam = cv.VideoCapture(0)
        ret, frame = cam.read()
        if not ret:
            return
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        H, W = frame.shape
        self.frame_stack = np.zeros(shape=(H, W, self.stack_length), dtype=np.uint8)

        ts1 = 0
        while True:
            ts0 = ts1
            ts1 = time.time()
            fps = 1/(ts1-ts0)
            self.stack_index = (self.stack_index + 1) % self.stack_length

            ret, frame = cam.read()
            if not ret:
                return
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            self.frame_stack[:, :, self.stack_index] = frame

            img = self.process()

            cv.putText(img, f"{fps: 4.0f} fps", (10, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,))
            cv.imshow("Video Spaced Delay", img)
            key = cv.waitKey(1)

        cam.release()
        cv.destroyAllWindows()

if __name__ == '__main__':
    vsd = VideoSpacialDelay()
    vsd.run()