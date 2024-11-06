import sys
import cv2
import pandas as pd
import os
import re  # For sanitizing filenames
from PyQt5.QtCore import Qt, QTimer, QPointF, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QLabel,
    QFileDialog,
    QHBoxLayout,
    QVBoxLayout,
    QLineEdit,
    QMessageBox,
    QSlider,
    QComboBox,
    QGroupBox,
    QListWidget,
    QListWidgetItem,
    QSplitter,
)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QPolygonF, QBrush
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPolygonItem, QGraphicsEllipseItem


class SegmentationWindow(QWidget):
    # Define a custom signal that emits frame number, mask path, and point coordinates
    mask_and_point_saved = pyqtSignal(int, str, tuple)

    def __init__(self, frame_image, frame_num, save_dir):
        super().__init__()
        self.setWindowTitle(f"Segmentation - Frame {frame_num}")
        self.frame_image = frame_image
        self.frame_num = frame_num
        self.save_dir = save_dir

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.pixmap = QPixmap.fromImage(self.frame_image)
        self.image_label.setPixmap(self.pixmap)

        # Graphics View for drawing
        self.graphics_view = QGraphicsView()
        self.scene = QGraphicsScene()
        self.graphics_view.setScene(self.scene)
        self.graphics_view.setRenderHint(QPainter.Antialiasing)
        self.graphics_view.setDragMode(QGraphicsView.NoDrag)

        self.polygon = QPolygonF()
        self.polygon_item = None
        self.point_item = None
        self.point = None
        self.drawing = False
        self.annotating_point = False  # Flag to indicate point annotation mode

        # Instructions
        self.instructions = QLabel("Left-click to add polygon vertices.\nRight-click to finish the polygon.")
        self.point_instructions = QLabel("Left-click to annotate a point on the mask.")

        # Buttons
        self.save_btn = QPushButton("Save Mask")
        self.save_btn.clicked.connect(self.save_mask)

        self.clear_btn = QPushButton("Clear Polygon")
        self.clear_btn.clicked.connect(self.clear_polygon)

        self.save_point_btn = QPushButton("Save Point")
        self.save_point_btn.setEnabled(False)
        self.save_point_btn.clicked.connect(self.save_point)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.save_btn)
        button_layout.addWidget(self.clear_btn)
        button_layout.addWidget(self.save_point_btn)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.instructions)
        layout.addWidget(self.graphics_view)
        layout.addWidget(self.point_instructions)
        layout.addLayout(button_layout)
        self.setLayout(layout)

        # Initialize scene with image
        self.scene.addPixmap(self.pixmap)
        self.setGeometry(150, 150, self.pixmap.width(), self.pixmap.height() + 150)

    def mousePressEvent(self, event):
        # Get position relative to graphics_view
        pos = self.graphics_view.mapFromParent(event.pos())
        scene_pos = self.graphics_view.mapToScene(pos)

        if self.annotating_point:
            if event.button() == Qt.LeftButton:
                self.point = (scene_pos.x(), scene_pos.y())
                self.draw_point(scene_pos)
                self.save_point_btn.setEnabled(True)
        else:
            # Ensure that the click is within the graphics view
            if not self.graphics_view.geometry().contains(event.pos()):
                return

            if event.button() == Qt.LeftButton:
                # Add polygon vertex
                self.polygon << scene_pos
                if self.polygon_item:
                    self.scene.removeItem(self.polygon_item)
                self.polygon_item = QGraphicsPolygonItem(QPolygonF(self.polygon))
                pen = QPen(QColor(255, 0, 0), 2)
                self.polygon_item.setPen(pen)
                self.scene.addItem(self.polygon_item)
            elif event.button() == Qt.RightButton and len(self.polygon) >= 3:
                # Finish polygon and prompt for point annotation
                self.drawing = False
                QMessageBox.information(self, "Polygon Completed", "Polygon drawing completed.\nNow, annotate a point on the mask.")
                self.annotating_point = True
                self.point_instructions.setText("Left-click to annotate a point on the mask.")
                self.save_point_btn.setEnabled(False)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()

    def save_mask(self):
        if len(self.polygon) < 3:
            QMessageBox.warning(self, "No Polygon", "Please draw a polygon before saving.")
            return

        # Create mask
        mask = QImage(self.pixmap.size(), QImage.Format_ARGB32)
        mask.fill(Qt.transparent)

        painter = QPainter(mask)
        painter.setBrush(QColor(255, 255, 255, 255))
        painter.setPen(Qt.NoPen)
        polygon = QPolygonF(self.polygon)
        painter.drawPolygon(polygon)
        painter.end()

        # Save mask
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        mask_path = os.path.join(self.save_dir, f"frame_{self.frame_num}_mask.png")
        mask.save(mask_path)

        QMessageBox.information(self, "Mask Saved", f"Mask saved to {mask_path}.\nPlease annotate a point on the mask.")

        # Enable point annotation
        self.annotating_point = True
        self.point_instructions.setText("Left-click to annotate a point on the mask.")
        self.save_point_btn.setEnabled(False)

    def draw_point(self, scene_pos):
        # Draw a small red circle to represent the point
        radius = 5
        pen = QPen(QColor(0, 255, 0))
        brush = QBrush(QColor(0, 255, 0))
        self.point_item = QGraphicsEllipseItem(scene_pos.x() - radius, scene_pos.y() - radius, 2 * radius, 2 * radius)
        self.point_item.setPen(pen)
        self.point_item.setBrush(brush)
        self.scene.addItem(self.point_item)

    def save_point(self):
        if self.point:
            # Emit the mask_saved signal with frame_num, mask_path, and point
            mask_path = os.path.join(self.save_dir, f"frame_{self.frame_num}_mask.png")
            self.mask_and_point_saved.emit(self.frame_num, mask_path, self.point)
            QMessageBox.information(self, "Point Saved", f"Point annotated at ({self.point[0]:.2f}, {self.point[1]:.2f}).")
            self.close()
        else:
            QMessageBox.warning(self, "No Point", "Please annotate a point before saving.")

    def clear_polygon(self):
        if self.polygon_item:
            self.scene.removeItem(self.polygon_item)
            self.polygon_item = None
        if self.point_item:
            self.scene.removeItem(self.point_item)
            self.point_item = None
        self.polygon = QPolygonF()
        self.point = None
        self.annotating_point = False
        self.save_point_btn.setEnabled(False)
        self.point_instructions.setText("Left-click to add polygon vertices.\nRight-click to finish the polygon.")


class VideoTagger(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Advanced Video Tagger with Segmentation and Point Annotation")
        self.setGeometry(50, 50, 1300, 900)

        # Video related variables
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.is_paused = True
        self.frame_num = 0
        self.fps = 0
        self.total_frames = 0
        self.duration = 0

        # Tagging data
        self.tags = []

        # Directory to save masks
        self.masks_dir = None  # Will be set dynamically based on video name

        # Directory to save clips
        self.clips_dir = None  # Will be set dynamically based on video name

        # Video path
        self.video_path = None  # To store the path of the loaded video

        # Flag to indicate if the slider is being dragged
        self.slider_is_dragging = False

        # Reference to segmentation window to prevent garbage collection
        self.segmentation_window = None

        # Setup UI
        self.setup_ui()

    def setup_ui(self):
        # Video display label
        self.video_label = QLabel("Load a video to start.")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("QLabel { background-color : black; }")
        self.video_label.setFixedSize(800, 450)  # Increased size for better visibility

        # Load button
        self.load_btn = QPushButton("Load Video")
        self.load_btn.clicked.connect(self.load_video)

        # Play/Pause button
        self.play_pause_btn = QPushButton("Play")
        self.play_pause_btn.setEnabled(False)
        self.play_pause_btn.clicked.connect(self.play_pause_video)

        # Tagging Group Box
        tag_group = QGroupBox("Add Tag")
        tag_layout = QHBoxLayout()

        # Object Name Input
        self.object_input = QLineEdit()
        self.object_input.setPlaceholderText("Enter object name...")
        self.object_input.setEnabled(False)

        # Action Selection
        self.action_combo = QComboBox()
        self.action_combo.addItems(["Select Action", "Start", "End", "Annotate"])
        self.action_combo.setEnabled(False)

        # Tag Button
        self.tag_btn = QPushButton("Add Tag")
        self.tag_btn.setEnabled(False)
        self.tag_btn.clicked.connect(self.add_tag)

        tag_layout.addWidget(QLabel("Object Name:"))
        tag_layout.addWidget(self.object_input)
        tag_layout.addWidget(QLabel("Action:"))
        tag_layout.addWidget(self.action_combo)
        tag_layout.addWidget(self.tag_btn)

        tag_group.setLayout(tag_layout)

        # Save button
        self.save_btn = QPushButton("Save Tags")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_tags)

        # Split Video button
        self.split_btn = QPushButton("Split Video")
        self.split_btn.setEnabled(False)
        self.split_btn.clicked.connect(self.split_video)

        # Progress Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(0)
        self.slider.sliderPressed.connect(self.slider_pressed)
        self.slider.sliderReleased.connect(self.slider_released)
        self.slider.sliderMoved.connect(self.slider_moved)

        # Current Time Label
        self.current_time_label = QLabel("00:00")
        self.current_time_label.setFixedWidth(60)
        self.current_time_label.setAlignment(Qt.AlignCenter)

        # Total Duration Label
        self.total_time_label = QLabel("00:00")
        self.total_time_label.setFixedWidth(60)
        self.total_time_label.setAlignment(Qt.AlignCenter)

        # Tag List for Start and Annotate Tags
        self.tag_list = QListWidget()
        self.tag_list.setEnabled(False)
        self.tag_list.itemDoubleClicked.connect(self.open_segmentation)

        tag_list_layout = QVBoxLayout()
        tag_list_layout.addWidget(QLabel("Start & Annotate Tags:"))
        tag_list_layout.addWidget(self.tag_list)

        # Layouts
        control_layout = QHBoxLayout()
        control_layout.addWidget(self.load_btn)
        control_layout.addWidget(self.play_pause_btn)
        control_layout.addStretch()
        control_layout.addWidget(self.save_btn)
        control_layout.addWidget(self.split_btn)

        slider_layout = QHBoxLayout()
        slider_layout.addWidget(self.current_time_label)
        slider_layout.addWidget(self.slider)
        slider_layout.addWidget(self.total_time_label)

        # Splitter to divide video display and tag list
        splitter = QSplitter(Qt.Horizontal)
        video_layout = QVBoxLayout()
        video_layout.addWidget(self.video_label)
        video_layout.addLayout(slider_layout)
        video_widget = QWidget()
        video_widget.setLayout(video_layout)
        splitter.addWidget(video_widget)

        tag_list_widget = QWidget()
        tag_list_widget.setLayout(tag_list_layout)
        splitter.addWidget(tag_list_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        main_layout = QVBoxLayout()
        main_layout.addWidget(splitter)
        main_layout.addWidget(tag_group)
        main_layout.addLayout(control_layout)

        self.setLayout(main_layout)

    def load_video(self):
        video_path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                QMessageBox.critical(self, "Error", "Failed to open the video.")
                return

            # Get video properties
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.duration = self.total_frames / self.fps

            self.frame_num = 0
            self.tags = []

            # Store video path
            self.video_path = video_path

            # Extract video name without extension
            video_name = os.path.splitext(os.path.basename(video_path))[0]

            # Set masks_dir to 'masks/video_name/'
            self.masks_dir = os.path.join('masks', video_name)
            if not os.path.exists(self.masks_dir):
                os.makedirs(self.masks_dir)

            # Set clips_dir to 'clips/video_name/'
            self.clips_dir = os.path.join('clips', video_name)
            if not os.path.exists(self.clips_dir):
                os.makedirs(self.clips_dir)

            # Enable buttons and slider
            self.play_pause_btn.setEnabled(True)
            self.object_input.setEnabled(True)
            self.action_combo.setEnabled(True)
            self.tag_btn.setEnabled(True)
            self.save_btn.setEnabled(True)
            self.split_btn.setEnabled(True)
            self.slider.setEnabled(True)
            self.tag_list.setEnabled(True)
            self.tag_list.clear()

            self.slider.setMaximum(self.total_frames - 1)

            # Update total time label
            total_minutes = int(self.duration // 60)
            total_seconds = int(self.duration % 60)
            self.total_time_label.setText(f"{total_minutes:02}:{total_seconds:02}")

            # Reset video to start
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.display_frame()
            self.play_pause_btn.setText("Play")
            self.is_paused = True

    def play_pause_video(self):
        if self.is_paused:
            self.timer.start(int(1000 / self.fps))
            self.play_pause_btn.setText("Pause")
        else:
            self.timer.stop()
            self.play_pause_btn.setText("Play")
        self.is_paused = not self.is_paused

    def next_frame(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.frame_num += 1
                self.display_frame(frame)
                self.update_slider()
            else:
                self.timer.stop()
                self.play_pause_btn.setText("Play")
                self.is_paused = True
                self.cap.release()

    def display_frame(self, frame=None):
        if frame is None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                return
            self.frame_num += 1

        if frame is not None:
            self.current_frame = frame.copy()
            # Convert the image from BGR (OpenCV) to RGB (Qt)
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = rgb_image.shape
            bytes_per_line = 3 * width
            qt_image = QImage(
                rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888
            )
            scaled_image = qt_image.scaled(
                self.video_label.width(),
                self.video_label.height(),
                Qt.KeepAspectRatio,
            )
            self.video_label.setPixmap(QPixmap.fromImage(scaled_image))
            self.current_display_image = qt_image

    def add_tag(self):
        if not self.cap.isOpened():
            return

        object_name = self.object_input.text().strip()
        action = self.action_combo.currentText()

        if not object_name:
            QMessageBox.warning(self, "Input Error", "Please enter an object name.")
            return

        if action not in ["Start", "End", "Annotate"]:
            QMessageBox.warning(
                self, "Input Error", "Please select a valid action (Start/End/Annotate)."
            )
            return

        # Get current timestamp
        current_time = self.frame_num / self.fps

        # Store tag data
        tag_data = {
            "Frame": self.frame_num,
            "Time (s)": round(current_time, 2),
            "Object Name": object_name,
            "Action": action,
            "Mask Path": "",       # To be updated if segmentation is done
            "HandTip": "",         # To be updated if point is annotated
            # "Point Y": "",         # To be updated if point is annotated
        }
        self.tags.append(tag_data)

        # If action is Start or Annotate, add to tag list
        if action in ["Start", "Annotate"]:
            item = QListWidgetItem(f"Frame {self.frame_num}: {object_name}")
            self.tag_list.addItem(item)

        QMessageBox.information(
            self,
            "Tag Added",
            f"Tag added:\nObject: {object_name}\nAction: {action}\nTime: {self.format_time(current_time)}.",
        )

        # Clear the inputs
        self.object_input.clear()
        self.action_combo.setCurrentIndex(0)

    def save_tags(self):
        if not self.tags:
            QMessageBox.warning(self, "No Tags", "No tags to save.")
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Tags", "", "CSV Files (*.csv)"
        )
        if save_path:
            df = pd.DataFrame(self.tags)
            df.to_csv(save_path, index=False)
            QMessageBox.information(self, "Saved", f"Tags saved to {save_path}.")

    def split_video(self):
        if not self.cap or not self.cap.isOpened():
            QMessageBox.warning(self, "Error", "No video loaded.")
            return

        if not self.tags:
            QMessageBox.warning(self, "No Tags", "No tags available to split.")
            return

        # Extract video name
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]

        # Set clips_dir to 'clips/video_name/'
        clips_dir = self.clips_dir
        if not os.path.exists(clips_dir):
            os.makedirs(clips_dir)

        # Sort tags by frame number
        sorted_tags = sorted(self.tags, key=lambda x: x['Frame'])

        # Pair Start and End tags
        pairs = []
        stack = []
        for tag in sorted_tags:
            if tag['Action'] == 'Start':
                stack.append(tag)
            elif tag['Action'] == 'End':
                if stack:
                    start_tag = stack.pop(0)  # FIFO for matching
                    if tag['Frame'] > start_tag['Frame']:
                        pairs.append((start_tag, tag))
                    else:
                        QMessageBox.warning(
                            self, "Tag Pair Error",
                            f"End tag at frame {tag['Frame']} precedes Start tag at frame {start_tag['Frame']}."
                        )
                else:
                    QMessageBox.warning(self, "Unmatched End Tag",
                                        f"End tag at frame {tag['Frame']} without matching Start tag.")

        if stack:
            QMessageBox.warning(self, "Unmatched Start Tags",
                                f"{len(stack)} Start tag(s) without matching End tag.")

        if not pairs:
            QMessageBox.warning(self, "No Valid Pairs",
                                "No valid Start-End tag pairs found to split.")
            return

        # Initialize VideoCapture for splitting
        cap_split = cv2.VideoCapture(self.video_path)
        if not cap_split.isOpened():
            QMessageBox.critical(self, "Error", "Failed to open the video for splitting.")
            return

        # Get video properties
        fps = cap_split.get(cv2.CAP_PROP_FPS)
        width = int(cap_split.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap_split.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec = cv2.VideoWriter_fourcc(*'mp4v')  # You can change codec if needed

        # Dictionary to keep track of clip counts per object
        object_clip_counts = {}

        # Iterate through pairs and save clips
        for idx, (start_tag, end_tag) in enumerate(pairs, start=1):
            start_frame = start_tag['Frame']
            end_frame = end_tag['Frame']
            object_name = start_tag['Object Name']

            # Sanitize object name for filename
            sanitized_object_name = self.sanitize_filename(object_name)

            # Update clip count for the object
            if sanitized_object_name in object_clip_counts:
                object_clip_counts[sanitized_object_name] += 1
            else:
                object_clip_counts[sanitized_object_name] = 1
            clip_idx = object_clip_counts[sanitized_object_name]

            # Define clip filename
            clip_filename = f"{sanitized_object_name}_clip_{clip_idx}.mp4"
            clip_path = os.path.join(clips_dir, clip_filename)

            # Initialize VideoWriter
            writer = cv2.VideoWriter(clip_path, codec, fps, (width, height))

            # Set VideoCapture to start_frame
            cap_split.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            # Read and write frames from start_frame to end_frame
            for frame_num in range(start_frame, end_frame + 1):
                ret, frame = cap_split.read()
                if not ret:
                    QMessageBox.warning(self, "Error",
                                        f"Failed to read frame {frame_num}.")
                    break
                writer.write(frame)

            writer.release()

        cap_split.release()

        QMessageBox.information(
            self, "Success",
            f"Video split into {len(pairs)} clip(s) in '{clips_dir}'."
        )

    def sanitize_filename(self, name):
        """
        Sanitize the object name to create a safe filename.
        Removes invalid characters and replaces spaces with underscores.
        """
        # Replace spaces with underscores
        name = name.replace(' ', '_')
        # Remove any character that is not alphanumeric or underscore
        name = re.sub(r'[^\w\-]', '', name)
        return name

    def update_slider(self):
        if not self.slider_is_dragging:
            self.slider.blockSignals(True)
            self.slider.setValue(self.frame_num)
            self.slider.blockSignals(False)

            # Update current time label
            current_time = self.frame_num / self.fps
            self.current_time_label.setText(self.format_time(current_time))

    def format_time(self, seconds):
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02}:{secs:02}"

    def slider_pressed(self):
        self.slider_is_dragging = True
        self.timer.stop()
        self.play_pause_btn.setText("Play")
        self.is_paused = True

    def slider_released(self):
        self.slider_is_dragging = False
        desired_frame = self.slider.value()
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, desired_frame)
            ret, frame = self.cap.read()
            if ret:
                self.frame_num = desired_frame
                self.display_frame(frame)
                self.update_slider()

        # Optionally resume playing after seeking
        # Uncomment the following lines if you want to resume playback after seeking
        # if not self.is_paused:
        #     self.timer.start(int(1000 / self.fps))

    def slider_moved(self, position):
        if self.cap.isOpened():
            desired_frame = position
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, desired_frame)
            ret, frame = self.cap.read()
            if ret:
                self.frame_num = desired_frame
                self.display_frame(frame)
                self.update_slider()

    def open_segmentation(self, item):
        # Find the tag corresponding to the selected item
        tag_text = item.text()
        frame_num = int(tag_text.split(":")[0].split()[1])
        tag_index = next(
            (i for i, tag in enumerate(self.tags) if tag["Frame"] == frame_num), None
        )
        if tag_index is None:
            QMessageBox.warning(self, "Error", "Tag not found.")
            return

        # Retrieve the frame image
        if self.cap.isOpened():
            current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = self.cap.read()
            if ret:
                # Convert frame to QImage
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, channel = rgb_image.shape
                bytes_per_line = 3 * width
                qt_image = QImage(
                    rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888
                )
                # Open segmentation window
                self.segmentation_window = SegmentationWindow(
                    qt_image, frame_num, self.masks_dir
                )
                self.segmentation_window.mask_and_point_saved.connect(self.update_mask_and_point)
                self.segmentation_window.show()
            else:
                QMessageBox.warning(self, "Error", "Failed to retrieve the frame.")
        else:
            QMessageBox.warning(self, "Error", "Video is not loaded.")

    def update_mask_and_point(self, frame_num, mask_path, point):
        # Update the tag's mask path and point coordinates
        for tag in self.tags:
            if tag["Frame"] == frame_num:
                tag["Mask Path"] = mask_path
                tag["HandTip"] = (round(point[0], 2),round(point[1], 2))
                # tag["Point Y"] = 
        QMessageBox.information(
            self,
            "Annotation Updated",
            f"Mask and point saved for frame {frame_num}."
        )

    def closeEvent(self, event):
        # Ensure that the video capture is released when the application is closed
        if self.cap and self.cap.isOpened():
            self.cap.release()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = VideoTagger()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()