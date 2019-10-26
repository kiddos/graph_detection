import sys
import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model, Sequential

from PyQt5.QtCore import pyqtSignal as Signal
from PyQt5.QtCore import QObject
from PyQt5.QtCore import QRectF
from PyQt5.QtCore import QPointF
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtGui import QPainter
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPen
from PyQt5.QtGui import QColor

from graph_ui import Ui_MainWindow as MainWindow


tf.enable_eager_execution()

meta_size = (800, 400)

def load_model():
  current_dir = os.path.dirname(os.path.relpath(__file__))
  model_path = os.path.join(current_dir, 'models', 'graph_detection.h5')
  if os.path.isfile(model_path):
    model = tf.keras.models.load_model(model_path)
    return model
  return None


class ControlWidget(QWidget):
  def __init__(self, root, params):
    QWidget.__init__(self, root)
    self.image = None
    self.control_radius = 10
    self.current_area = None
    self.area = []
    self.mode = 'add'
    self.params = params
    self.cropped = []

    self.running_prediction = False
    self.model = load_model()

  def draw_area(self, painter, area):
    painter.begin(self)
    if self.mode == 'add':
      painter.setPen(QColor(39, 174, 96))
    if self.mode == 'edit':
      painter.setPen(QColor(203, 67, 53))

    w = self.width()
    h = self.height() / 2.0
    scaled = [QPointF(p.x() * w, p.y() * h) for p in area]
    painter.drawPolygon(*scaled)
    painter.end()

  def draw_grid(self, painter):
    w = float(self.width() - 2)
    h = float(self.height())
    painter.setPen(QColor(0, 0, 0))
    for i in range(5):
      painter.drawLine(QPointF(0, i * h / 8 + h / 2), QPointF(w, i * h / 8 + h / 2))
    for j in range(5):
      painter.drawLine(QPointF(j * w / 4, h / 2), QPointF(j * w / 4, 3 * h / 8 + h / 2))

  def to_binary(self, cropped):
    threshold = cropped.mean() * self.params['threshold'] / 100.0
    c1 = cropped[:, :, 0] < threshold
    c2 = cropped[:, :, 1] < threshold
    c3 = cropped[:, :, 2] < threshold
    filtered = np.logical_and(np.logical_and(c1, c2), c3).astype(np.uint8) * 255
    return filtered

  def find_sample_points(self, cropped):
    binary = self.to_binary(cropped)
    data = []
    height = binary.shape[0]
    for x in range(binary.shape[1]):
      col = binary[:, x]
      if col.max() > 0:
        # reverse: find the points that is lowest (maximum y)
        v = np.argmax(col[::-1])
        y = height - v
        data.append([x, y])
    return np.array(data, dtype=np.int32)

  def compute_base_line(self, points):
    # trapazoidal rule
    total_t = 0
    total_integral = 0
    for i in range(len(points) - 1):
      t = points[i+1, 0] - points[i, 0]
      integral = (points[i+1, 1] + points[i, 1]) * t / 2
      total_integral += integral
      total_t += t
    return total_integral / total_t

  def draw_info(self, painter):
    w = float(self.width())
    h = float(self.height())

    #  painter.drawText(10, h / 2, text_w, text_h, Qt.AlignTop | Qt.AlignLeft,
    #      'cell = %.3f (mV)' % (self.params['mv']))

    #  painter.drawText(0, h / 8 + h / 2 - text_h, text_w, text_h,
    #      Qt.AlignTop | Qt.AlignLeft,
    #      'min = %.3f (mV)' % (self.params['min']))

    font = painter.font()
    font.setPointSizeF(8.0)
    painter.setFont(font)
    for i, c in enumerate(self.cropped):
      text_w, text_h = 200, 20
      cropped_w = float(c.shape[1])
      cropped_h = float(c.shape[0])
      sec = cropped_w / (cropped_h / 4.0) * self.params['sec']
      text = 'period = %.3f (s)' % (sec)

      area_width = w / 4
      area_height = h / 8
      if i == len(self.cropped) - 1:
        area_width = w

      loc_x = float(i % 4) * area_width + area_width / 2
      loc_y = float(i // 4) * area_height + h / 2 + area_height
      painter.setPen(QColor(203, 69, 53))
      painter.drawText(loc_x - text_w / 2, loc_y - text_h,
          text_w, text_h, Qt.AlignHCenter, text)

      point_color = QColor(93, 209, 91, 160)
      painter.setPen(point_color)
      painter.setBrush(point_color)
      sample_points = self.find_sample_points(c)
      # draw points
      size = 4
      for row in sample_points:
        px = row[0] / cropped_w * area_width + float(i % 4) * area_width
        py = row[1] / cropped_h * area_height + h / 2 + float(i // 4) * area_height
        painter.drawEllipse(px - size / 2, py - size / 2, size, size)

      # draw baseline
      painter.setPen(QColor(227, 79, 227))
      baseline = self.compute_base_line(sample_points)
      bx = float(i % 4) * area_width
      by = float(i // 4) * area_height + h / 2 + baseline / cropped_h * area_height
      painter.drawLine(bx, by, bx + area_width, by)

  def paintEvent(self, e):
    painter = QPainter()
    w = float(self.width())
    h = float(self.height())

    if self.image:
      painter.begin(self)
      painter.drawImage(QRectF(0.0, 0.0, w, h / 2.0), self.image)
      painter.end()

      for area in self.area:
        self.draw_area(painter, area)

    if hasattr(self, 'graph') and self.graph:
      painter.begin(self)
      painter.drawImage(QRectF(0.0, h / 2.0, w, h / 2.0), self.graph)

      self.draw_info(painter)
      self.draw_grid(painter)
      painter.end()

  def draw_image(self, fname):
    image = QImage()
    image.load(fname)
    self.image = image
    img = cv2.imread(fname)
    self.original = img
    img = cv2.resize(img.copy(), meta_size)
    self.img_data = img

    self.predict()
    self.repaint()

  def draw_sample_points(self, cropped):
    sample_points = self.find_sample_points(cropped)
    for row in sample_points:
      cv2.circle(cropped, (row[0], row[1]), 5, color=(0, 255, 0), thickness=-1)

  def to_display(self, cropped):
    display_image = np.zeros([meta_size[1], meta_size[0], 3], dtype=np.uint8)
    for i in range(3):
      for j in range(4):
        portion = cropped[i * 4 + j].copy()
        #  self.draw_sample_points(portion)
        portion = cv2.resize(portion, (200, 100))
        display_image[i*100:(i+1)*100, j*200:(j+1)*200] = portion

    portion = cropped[-1].copy()
    #  self.draw_sample_points(portion)
    portion = cv2.resize(portion, (meta_size[0], meta_size[1] // 4))
    display_image[300:, :] = portion

    display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
    return display_image

  def predict(self):
    self.running_prediction = True
    if hasattr(self, 'model') and self.model:
      self.area = []
      self.cropped = []
      inputs = self.img_data.reshape([-1, meta_size[1], meta_size[0], 3])
      p = self.model.predict(inputs.astype(np.float32))
      for row in p[0]:
        area = []
        for xy in row:
          area.append(QPointF(xy[0], xy[1]))
        self.area.append(area)

        self.cropped.append(self.crop(row))

      display_images = self.to_display(self.cropped)
      self.graph = QImage(display_images,
          display_images.shape[1], display_images.shape[0], QImage.Format_RGB888)
      self.repaint()

    self.running_prediction = False

  def crop(self, area):
    image_h, image_w = self.original.shape[:2]
    src = area * np.array([image_w, image_h], dtype=np.float32)
    w = int(np.max(src[:, 0]) - np.min(src[:, 0]))
    h = int(np.max(src[:, 1]) - np.min(src[:, 1]))

    dst = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]], dtype=np.float32)

    P = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(self.original.copy(), P, (w, h))
    return warped

  def set_threshold(self, value):
    self.params['threshold'] = value

    display_images = self.to_display(self.cropped)
    self.graph = QImage(display_images,
        display_images.shape[1], display_images.shape[0], QImage.Format_RGB888)
    self.repaint()

  def set_sec(self, value):
    try:
      self.params['sec'] = float(value)
      self.repaint()
    except Exception:
      pass

  def scale_data(self, sample_points, w, h):
    entries = []
    for row in sample_points:
      value = (h - row[1]) / h * 4 * self.params['mv']
      t = row[0] / (h / 4.0) * self.params['sec']
      entries.append([t, value])
    entries = np.array(entries)
    return entries


  def get_all_data(self):
    data = []
    for c in self.cropped:
      sample_points = self.find_sample_points(c)
      scaled = self.scale_data(sample_points, c.shape[1], c.shape[0])
      baseline = self.compute_base_line(scaled)
      scaled[:, 1] = scaled[:, 1] - baseline
      data.append(scaled)
    return data

  def save_data(self, data_file):
    try:
      data = self.get_all_data()
      with open(data_file, 'w') as f:
        for entries in data:
          for row in entries:
            f.write('%f,%f\n' % (row[0], row[1]))
          f.write('\n')
    except Exception:
      pass


class RootWindow(QObject):
  def __init__(self):
    QObject.__init__(self)
    self.root = QMainWindow()

    self.main = MainWindow()
    self.main.setupUi(self.root)

    self.setup_draw()

    self.main.btnOpenImage.clicked.connect(self.open_image)
    self.main.thresholdSlider.valueChanged.connect(self.control_widget.set_threshold)
    self.main.btnSave.clicked.connect(self.save_data)
    self.main.leMV.textChanged[str].connect(self.handle_mv_change)
    self.main.leSec.textChanged[str].connect(self.control_widget.set_sec)
    self.main.actionFileOpen.triggered.connect(self.open_image)

  def setup_draw(self):
    layout = QVBoxLayout(self.main.imagePanel)
    params = {
      'mv': float(self.main.leMV.text()),
      'sec': float(self.main.leSec.text()),
      'threshold': float(self.main.thresholdSlider.value())
    }
    self.control_widget = ControlWidget(self.main.imagePanel, params)
    layout.addWidget(self.control_widget)
    self.main.imagePanel.setLayout(layout)

  def show(self):
    self.root.show()

  def set_param(self, key, value_str):
    if len(value_str) > 0:
      try:
        self.control_widget.params[key] = float(value_str)
        self.control_widget.repaint()
      except Exception:
        pass

  def handle_mv_change(self, value):
    self.set_param('mv', value)

  def open_image(self):
    fname, _ = QFileDialog.getOpenFileName(self.root, 'Open file',
        '', "Image files (*.jpg *.gif)")
    if len(fname) > 0 and (fname.lower().endswith('.jpg') or fname.lower().endswith('.png')):
      self.control_widget.draw_image(fname)

  def save_data(self):
    folder = QFileDialog.getExistingDirectory(self.root, 'Select a folder:', '',
      QFileDialog.ShowDirsOnly)

    if folder:
      self.control_widget.save_data(os.path.join(folder, 'data.csv'))


def main():
  app = QApplication(sys.argv)
  root = RootWindow()
  root.show()
  app.exec_()


if __name__ == '__main__':
  main()
