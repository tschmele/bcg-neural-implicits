import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import time

# --- Simple drawing widget for testing the model ---
# This is a simple drawing widget that can be used to test the model.
# It creates a window with a black background which can be drawn on by clicking and dragging the mouse.
# The key 'c' clears the drawing and the key 'q' quits the program.
# A bar plot shows the current class probablities predicted by the model.
class DrawWindow:
    def __init__(
            self,
            window_name = "Drawy McDrawface",       # name of the window
            on_change_callback = None,              # callback function that is called when the drawing changes. It has the signature (image) -> class_probabilities.
            num_classes = 1,                        # number of classes
            class_names = ["C"],                    # names of the classes
            window_size = (512, 512),               # size of the drawing window
            brush_size = 10,                        # size of the brush
            channels = 1,                           # number of channels in the image
            dtype = np.float32,                     # data type of the image
            plot_update_interval = 100              # update interval for the probability plot in ms
        ):
        # options
        self.brush_size = brush_size
        self.window_size = window_size
        self.window_name = window_name
        self.on_change_callback = on_change_callback
        self.num_classes = num_classes
        self.plot_update_interval = plot_update_interval # ms
        # state
        self.prev_x = -1
        self.prev_y = -1
        self.drawing = False
        self.current_class_probabilities = np.zeros(num_classes, dtype = dtype)
        self.plot_update_accum = 0.0
        self.last_t = 0.0
        # image
        self.image = np.zeros((self.window_size[0], self.window_size[1]) if channels == 1 else (self.window_size[0], self.window_size[1], channels), dtype)
        # figure and plot for plotting class probabilities
        self.class_names = class_names
        self.fig = plt.figure(self.window_name)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_ylim([0, 1])
        self.ax.set_xlabel("Class")
        self.ax.set_ylabel("Probability")
        self.barplot = self.ax.bar(self.class_names, self.current_class_probabilities)

        # create window and set mouse callback
        self.window = cv.namedWindow(self.window_name, cv.WINDOW_AUTOSIZE)
        cv.setMouseCallback(self.window_name, self.mouse_callback)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.prev_x = x
            self.prev_y = y
        elif event == cv.EVENT_LBUTTONUP:
            self.drawing = False
            self.prev_x = x
            self.prev_y = y
        elif event == cv.EVENT_MOUSEMOVE:
            if self.drawing:
                self.draw_segment(self.prev_x, self.prev_y, x, y)
                self.prev_x = x
                self.prev_y = y

    def draw_segment(self, x0, y0, x1, y1):
        cv.line(self.image, (x0, y0), (x1, y1), (1.0, 1.0, 1.0), self.brush_size)
        cv.imshow(self.window_name, self.image)
        if self.on_change_callback is not None:
            self.current_class_probabilities = self.on_change_callback(self.image)
            # update plot every plot_update_interval ms
            t = time.time()
            self.plot_update_accum += (t - self.last_t) * 1000.0
            self.last_t = t
            if self.plot_update_accum > self.plot_update_interval:
                self.plot_class_probabilities()
                self.plot_update_accum = 0.0

    def plot_class_probabilities(self):
        for rect,h in zip(self.barplot, self.current_class_probabilities):
            rect.set_height(h)
        self.fig.canvas.draw()

    def clear(self):
        self.image = np.zeros_like(self.image)
        cv.imshow(self.window_name, self.image)
        self.current_class_probabilities = np.zeros(self.num_classes, dtype = self.current_class_probabilities.dtype)
        self.plot_class_probabilities()

    def show(self, wait = 0):
        # initialize plot
        self.plot_class_probabilities()
        self.fig.show()
        # show window
        cv.imshow(self.window_name, self.image)
        while(True):            
            k = chr(cv.waitKey(0))
            if k == 'q':
                cv.destroyAllWindows()
                return
            elif k == 'c':
                self.clear()