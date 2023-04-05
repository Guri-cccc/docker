#!/usr/bin/env python3

from __future__ import print_function

import sys

sys.path.append('/home/catkin_ws/src/button_recognition/scripts/ocr_rcnn_lib')

import rospy
import rosnode
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from button_recognition.msg import Button, ButtonArray

import cv2
import PIL.Image
import PIL.ImageOps
import numpy as np

from ocr_rcnn_lib.button_detection import ButtonDetector
from ocr_rcnn_lib.character_recognition import CharacterRecognizer
from ocr_rcnn_lib.button_recognition import ButtonRecognizer


class ButtonRecognitionInterence(object):
    def __init__(self):
        super(ButtonRecognitionInterence, self).__init__()
        rospy.init_node('train_automation')

        self.detector = ButtonDetector()
        self.recognizer = CharacterRecognizer()
        self.button_recognizer = ButtonRecognizer()

        self.sub_image = rospy.Subscriber("/cam_ee/color/image_raw", Image, self.ImageCallback)
        
        self.pub_detection_image = rospy.Publisher('/button_detection/image', Image, queue_size=10)
        self.pub_detection_result = rospy.Publisher('/button_detection/detections', ButtonArray, queue_size=10)

    def ImageCallback(self, msg):

        image_np = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        # if image_np.shape != (480, 640):
        #     img_pil = PIL.Image.fromarray(image_np)
        # delta_w, delta_h= 640 - img_pil.size[0], 480 - img_pil.size[1]
        # padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        # new_im = PIL.ImageOps.expand(img_pil, padding)
        # image_np = np.copy(np.asarray(new_im))
      
        # recognitions = self.button_recognizer.predict(image_np, False)
        # m_detection_result = ButtonArray()
        # for item in recognitions:
        #     button = Button()
        #     button.y_min = int(item[0][0] * image_np.shape[0])
        #     button.x_min = int(item[0][1] * image_np.shape[1])
        #     button.y_max = int(item[0][2] * image_np.shape[0])
        #     button.x_max = int(item[0][3] * image_np.shape[1])
        #     button.score = item[1] # float
        #     button.text = item[2]
        #     button.belief = item[3]
        #     m_detection_result.buttons.append(button)
        #     cv2.rectangle(image_np, (button.x_min, button.y_min), (button.x_max, button.y_max), (0, 255, 0), thickness=10)
            
        boxes, scores, _ = self.detector.predict(image_np, True)
        button_patches, button_positions, _ = self.button_candidates(boxes, scores, image_np)

        m_detection_result = ButtonArray()
        for button_img, button_pos in zip(button_patches, button_positions):
            button_text, button_score, button_draw =self.recognizer.predict(button_img, draw=True)
            x_min, y_min, x_max, y_max = button_pos
            
            button = Button()
            button.y_min = y_min
            button.x_min = x_min
            button.y_max = y_max
            button.x_max = x_max
            button.score = button_score
            button.text = button_text
            button.belief = button_score
            m_detection_result.buttons.append(button)
            
            text_position = (int((x_min+x_max)/2), int((y_min+y_max)/2))
            score_position = (int(x_min), int(y_min))
            cv2.rectangle(image_np, (button.x_min, button.y_min), (button.x_max, button.y_max), (0, 255, 0), thickness=3)
            cv2.putText(image_np, button_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image_np, str(round(button_score,3)), score_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        ros_image = Image()
        ros_image.data = image_np.flatten().tobytes()
        ros_image.header.stamp = rospy.Time.now()
        ros_image.header.frame_id = msg.header.frame_id
        ros_image.width = image_np.shape[1]
        ros_image.height = image_np.shape[0]
        ros_image.encoding = "bgr8"
        ros_image.step = image_np.shape[1] * image_np.shape[2]
        
        self.pub_detection_image.publish(ros_image)
        self.pub_detection_result.publish(m_detection_result)
        
        # self.button_recognizer.clear_session()
        
    def button_candidates(self, boxes, scores, image):
        img_height = image.shape[0]
        img_width = image.shape[1]

        button_scores = []
        button_patches = []
        button_positions = []

        for box, score in zip(boxes, scores):
            if score < 0.5:
                continue

            y_min = int(box[0] * img_height)
            x_min = int(box[1] * img_width)
            y_max = int(box[2] * img_height)
            x_max = int(box[3] * img_width)

            button_patch = image[y_min: y_max, x_min: x_max]
            button_patch = cv2.resize(button_patch, (180, 180))

            button_scores.append(score)
            button_patches.append(button_patch)
            button_positions.append([x_min, y_min, x_max, y_max])
        return button_patches, button_positions, button_scores

def main():

    button_detector = ButtonRecognitionInterence()
    rate = rospy.Rate(100)

    while not rospy.is_shutdown():
        rate.sleep()

if __name__ == '__main__':
    main()
