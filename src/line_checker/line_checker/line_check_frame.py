#line check.ipynb 에 있는 함수중 필요한 것만 가지고 옴


# Imports
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import time
import warnings
import matplotlib.pyplot as plt
# Dont show warnings
warnings.filterwarnings("ignore")

src = np.array([[598, 448], [684, 448], [1026, 668], [278, 668]], np.float32)
dst = np.array([[300, 0], [980, 0], [980, 720], [300, 720]], np.float32)

kernel_small = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], 'uint8')
# Convert image to yellow and white color space
#hsv를 통해 흰색과 노란색만 남기기
def color_space(img):
    # Convert image to HSV
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # Colorspace "yellow" in HSV: (15-40, 80-255, 160-255)
    mask_yellow = cv.inRange(img_hsv, (15, 100, 160), (40, 255, 255))
    # Colorspace "white" in HSV: (0-255, 0-20, 200-255)
    mask_white = cv.inRange(img_hsv, (0, 0, 200), (255, 70, 255))
    # Merge white and yellow masks
    masks = cv.bitwise_or(mask_yellow, mask_white)
    # Return image in gray
    
    return cv.cvtColor(cv.bitwise_and(img, img, mask=masks), cv.COLOR_BGR2GRAY)

#hls를 통해 흰색과 노란색만 남기기
def color_space_hls(img):
    # Convert image to HSV
    img_hls = cv.cvtColor(img, cv.COLOR_BGR2HLS)
    # Colorspace "yellow" in HSV: (15-40, 80-255, 160-255)
    mask_yellow = cv.inRange(img_hls, (15, 100, 0), (25, 150, 255))
    # Colorspace "white" in HSV: (0-255, 0-20, 200-255)
    mask_white = cv.inRange(img_hls, (0, 200, 0), (180, 255, 255))




    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # 초록색 제거용 마스크 (선택 사항)
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    green_mask = cv.inRange(hsv, lower_green, upper_green)


    # 노랑에서 초록 제거
    yellow_clean = cv.bitwise_and(mask_yellow, cv.bitwise_not(green_mask))

    yellow_clean_masks = cv.bitwise_and(mask_yellow, yellow_clean)

    # Merge white and yellow masks
    masks = cv.bitwise_or(yellow_clean_masks, mask_white)
    # Return image in gray

    #cv.imshow('hls', cv.cvtColor(cv.bitwise_and(img, img, mask=masks), cv.COLOR_HLS2BGR))
    #cv.imshow('hls', cv.bitwise_and(img, img, mask=masks))
    
    return cv.cvtColor(cv.bitwise_and(img, img, mask=masks), cv.COLOR_BGR2GRAY)


import numpy as np
import cv2

#차선 감지용 클래스
class LaneTracker:
    #nwindows : 슬라이딩 윈도우의 갯수, margin : 탐지할때의 마진 값, minimum : 탐지할때의 최솟값
    def __init__(self, nwindows=9, margin=200, minimum=30):
        self.prev_left_fit = None
        self.prev_right_fit = None
        self.nwindows = nwindows
        self.margin = margin
        self.minimum = minimum
        self.dummy = None
        self.reset_F = False
        

    def reset(self):
        self.prev_left_fit = None
        self.prev_right_fit = None

    #다항식 리셋용
    #후술할 추적 방식에 리셋시 필요할 경우를 위해 왼쪽 차선이 오른쪽 차선과 교차하면 리셋되게 만듦
    def should_reset(self, left_fit, right_fit, warped_img):
        if left_fit is None or right_fit is None:
            #print("차선 인식 안됨")
            return True

        ploty = np.linspace(0, warped_img.shape[0]-1, warped_img.shape[0])
        left_fitx = np.polyval(left_fit, ploty)
        right_fitx = np.polyval(right_fit, ploty)

        
        y_eval = np.max(ploty)  # 가장 아래 지점에서의 곡률 평가
        threshold = 2000
        def calc_curvature(fit):
            A = fit[0]
            B = fit[1]
            return ((1 + (2*A*y_eval + B)**2)**1.5) / np.abs(2*A)

        left_curve = calc_curvature(left_fit)
        right_curve = calc_curvature(right_fit)

        #diff = np.abs(left_curve - right_curve)

        if abs(left_fit[0] - right_fit[0]) > 5.0e-03:
            return True
        

        # 1. 좌우 교차 판단
        if np.any(left_fitx >= right_fitx):

            return True
        
        width = warped_img.shape[1]

        bottom_half_left_fitx = left_fitx[(warped_img.shape[0] // 2):]
        bottom_half_right_fitx = right_fitx[(warped_img.shape[0] // 2):]
        if np.all(bottom_half_left_fitx > width // 2) or np.all(bottom_half_right_fitx < width // 2):
            return True
        # 2. 거리 기반 판단
        #영상별로 차선간 거리가 달라져서 쓰기에는 힘들것으로 보임
        """
        lane_width = right_fitx - left_fitx
        if np.any(lane_width < 200) or np.any(lane_width > 800):
            return True
        """
        # 3. 곡률 차이 판단 (선택사항)
        # left_curv = calc_curvature(left_fit, ploty)
        # right_curv = calc_curvature(right_fit, ploty)
        # if abs(left_curv - right_curv) > threshold:
        #     return True

    
    def remove_outliers(self, x, y, fit, threshold=30):
        residuals = np.abs(np.polyval(fit, y) - x)
        mask = residuals < threshold
        return x[mask], y[mask]
    
    def find_good_inds(self, nonzerox, nonzeroy, win_x_low, win_x_high, win_y_low, win_y_high):
        return ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
    
    #warped_img 2진 이미지를 넣으면 그것을 바탕으로 차선을 탐지
    #draw 가 True 라면 슬라이딩 윈도우 시각화 됨
    def update(self, warped_img, draw=False):

        #차선 데이터 상태 확인해서 안좋으면 차선 데이터 리셋 
        if self.should_reset(self.prev_left_fit, self.prev_right_fit, warped_img):
            #result = self.sliding_windows_visual(warped_img, draw)
            """
            if self.dummy is not None:
                plt.imshow(cv.cvtColor(self.dummy, cv.COLOR_BGR2RGB))
                plt.show()
            """
            #차선 데이터 없을때 차선 탐지
            result = self.sliding_windows_visual_central(warped_img, draw)
        else:
            #이전 차선 데이터를 바탕으로 차선 탐지
            #이전 차선 데이터를 바탕으로 차선을 탐지하다보니 한번 뒤틀리면 계속 뒤틀림
            #그래서 결과값을 확인하고 결과값 리셋을 해줌
            result = self.quick_search(warped_img, draw)

        #나온 결과값을 바탕으로 상태 안좋으면 결과값 리셋
        if self.should_reset(result["left"]["fit"], result["right"]["fit"], warped_img):
            self.reset_F = True
            result["left"]["fit"] = None
            result["right"]["fit"] = None
        else:
            self.reset_F = False
        # 상태 갱신
        self.prev_left_fit = result["left"]["fit"]
        self.prev_right_fit = result["right"]["fit"]
        self.dummy = result["image"]
        return result
    #sliding window
    #평범한 sliding_window 방식
    #중간에서부터 슬라이딩 윈도우를 찾는 sliding_windows_visual_central 를 사용하는데 해당 방식이 안되면 이 방식을 한번 더 적용함
    def sliding_windows_visual(self, warped_img, draw):
        # ▶ ROI 마스킹: 잘못된 영역 제거
        mask = np.ones_like(warped_img, dtype=np.uint8) * 255
        height, width = warped_img.shape
        cv.rectangle(mask, (0, height - 20), (width, height), 0, -1)

        
        warped_img = cv.bitwise_and(warped_img, mask)

        
        """
        plt.imshow(cv.cvtColor(warped_img, cv.COLOR_BGR2RGB))
        plt.show()
        """
        # ▶ 모폴로지 연산으로 노이즈 제거
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        warped_img = cv.morphologyEx(warped_img, cv.MORPH_OPEN, kernel)

        # ▶ 시각화용 이미지 생성
        out_img = np.dstack((warped_img, warped_img, warped_img)) * 255

        # ▶ 히스토그램 기반 시작점 계산
        histogram = np.sum(warped_img[warped_img.shape[0]//2:, :], axis=0)
        midpoint = histogram.shape[0] // 2
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        """
        plt.plot(histogram)
        plt.title("Lane pixel histogram (bottom half)")
        plt.show()
        """
        # ▶ 슬라이딩 윈도우 초기화
        window_height = warped_img.shape[0] // self.nwindows
        nonzero = warped_img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base

        left_lane_inds = []
        right_lane_inds = []

        max_pixels = warped_img.size  # 충분히 큰 공간 확보
        left_inds = np.zeros(max_pixels, dtype=np.int32)
        right_inds = np.zeros(max_pixels, dtype=np.int32)
        left_idx = 0
        right_idx = 0

        for window in range(self.nwindows):
            win_y_low = warped_img.shape[0] - (window + 1) * window_height
            win_y_high = warped_img.shape[0] - window * window_height

            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin

            if draw:
                cv.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0,255,0), 2)
                cv.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0,255,255), 2)

            good_left_inds = self.find_good_inds(nonzerox, nonzeroy, win_xleft_low, win_xleft_high, win_y_low, win_y_high)
            good_right_inds = self.find_good_inds(nonzerox, nonzeroy, win_xright_low, win_xright_high, win_y_low, win_y_high)

            if len(good_left_inds) > self.minimum:
                leftx_current = int(np.mean(nonzerox[good_left_inds], dtype=np.float32))
            if len(good_right_inds) > self.minimum:
                rightx_current = int(np.mean(nonzerox[good_right_inds], dtype=np.float32))

            # ✅ 보호 코드 추가: overflow 방지
            n_left = len(good_left_inds)
            n_right = len(good_right_inds)

            if left_idx + n_left > max_pixels:
                print("⚠️ left_inds 배열 초과 발생! 일부 데이터는 무시됩니다.")
                n_left = max_pixels - left_idx
            if right_idx + n_right > max_pixels:
                print("⚠️ right_inds 배열 초과 발생! 일부 데이터는 무시됩니다.")
                n_right = max_pixels - right_idx

            left_inds[left_idx:left_idx+n_left] = good_left_inds[:n_left]
            right_inds[right_idx:right_idx+n_right] = good_right_inds[:n_right]
            left_idx += n_left
            right_idx += n_right

        # 슬라이스
        left_lane_inds = left_inds[:left_idx]
        right_lane_inds = right_inds[:right_idx]


        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # ▶ 이상점 제거
        left_fit, right_fit = None, None
        if len(leftx) > 20:
            #polyfit : 주어진 x,y 데이터를 바탕으로 다항식을 구해주는 함수 여기서는 마지변수가 2여서 2차함수
            left_fit = np.polyfit(lefty, leftx, 2)
            leftx, lefty = self.remove_outliers(leftx, lefty, left_fit)
            if len(leftx) > 0:
                left_fit = np.polyfit(lefty, leftx, 2)
            else:
                left_fit = None
        if len(rightx) > 20:
            right_fit = np.polyfit(righty, rightx, 2)
            rightx, righty = self.remove_outliers(rightx, righty, right_fit)
            if len(rightx) > 0:
                right_fit = np.polyfit(righty, rightx, 2)
            else:
                right_fit = None

        # ▶ 픽셀 색상 표시
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        # ▶ 보간 곡선 시각화
        if draw:
            ploty = np.linspace(0, warped_img.shape[0]-1, warped_img.shape[0])
            if left_fit is not None:
                left_fitx = np.polyval(left_fit, ploty)
                for i in range(len(ploty)-1):
                    cv.line(out_img, (int(left_fitx[i]), int(ploty[i])), (int(left_fitx[i+1]), int(ploty[i+1])), (255, 255, 0), 2)
            if right_fit is not None:
                right_fitx = np.polyval(right_fit, ploty)
                for i in range(len(ploty)-1):
                    cv.line(out_img, (int(right_fitx[i]), int(ploty[i])), (int(right_fitx[i+1]), int(ploty[i+1])), (0, 255, 255), 2)
        return {
            "image": out_img,
            "left": {
                "fit": left_fit,
                "x": leftx,
                "y": lefty,
            },
            "right": {
                "fit": right_fit,
                "x": rightx,
                "y": righty,
            }
        }
    #sliding window 중앙 기준으로
    #기본은 화면 끝에서부터 측정해서 멀리있는 차선이 인식되거나 할 경우 있음
    #중앙 기준의 경우 커브가 있어 중앙을 넘거나 중앙에 교통 마크가 있으면 문제 발생 가능성 있음
    #슬라이딩 윈도우는 아래에서부터 작은 화면을 통해 선을 추적하는 방식
    def sliding_windows_visual_central(self, warped_img_ori, draw):

        height, width = warped_img_ori.shape

        # ROI 마스킹
        mask = np.ones_like(warped_img_ori, dtype=np.uint8) * 255
        cv.rectangle(mask, (0, height - 50), (width, height), 0, -1)
        warped_img = cv.bitwise_and(warped_img_ori, mask)
        """
        plt.imshow(cv.cvtColor(warped_img, cv.COLOR_BGR2RGB))
        plt.show()
        """
        # 모폴로지 연산
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        warped_img = cv.morphologyEx(warped_img, cv.MORPH_OPEN, kernel)

        out_img = np.dstack((warped_img, warped_img, warped_img)) * 255

        # 중앙 기준 peak 검출
        histogram = np.sum(warped_img[height - height // 3:, :], axis=0)
        midpoint = width // 2
        threshold = 10000
        leftx_current = None
        rightx_current = None
        for i in range(midpoint, 0, -1):
            if histogram[i] > threshold:
                leftx_current = i
                break

        # 우측: 중앙에서 오른쪽으로 이동
        for i in range(midpoint, width):
            if histogram[i] > threshold:
                rightx_current = i
                break

        if  leftx_current == rightx_current:
            leftx_current = None
            rightx_current = None
            
        
        if  leftx_current == None:
            leftx_current = np.argmax(histogram[:midpoint])
        if rightx_current == None:
            rightx_current = np.argmax(histogram[midpoint:]) + midpoint

        """
        print(leftx_current)
        print(rightx_current)
        """
        """
        central_peak = np.argmax(histogram[midpoint - 100: midpoint + 100]) + (midpoint - 100)
        leftx_current = central_peak - (self.margin * 4)
        rightx_current = central_peak + (self.margin * 4)
        """
        """
        plt.plot(histogram)
        plt.title("Lane pixel histogram (bottom half)")
        plt.show()
        """
        window_height = height // self.nwindows
        nonzero = warped_img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_lane_inds = []
        right_lane_inds = []



        max_pixels = warped_img.size  # 충분히 큰 공간 확보
        left_inds = np.zeros(max_pixels, dtype=np.int32)
        right_inds = np.zeros(max_pixels, dtype=np.int32)
        left_idx = 0
        right_idx = 0

        for window in range(self.nwindows):
            win_y_low = warped_img.shape[0] - (window + 1) * window_height
            win_y_high = warped_img.shape[0] - window * window_height

            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin

            if draw:
                cv.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0,255,0), 2)
                cv.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0,255,255), 2)

            good_left_inds = self.find_good_inds(nonzerox, nonzeroy, win_xleft_low, win_xleft_high, win_y_low, win_y_high)
            good_right_inds = self.find_good_inds(nonzerox, nonzeroy, win_xright_low, win_xright_high, win_y_low, win_y_high)

            if len(good_left_inds) > self.minimum:
                leftx_current = int(np.mean(nonzerox[good_left_inds], dtype=np.float32))
            if len(good_right_inds) > self.minimum:
                rightx_current = int(np.mean(nonzerox[good_right_inds], dtype=np.float32))

            # ✅ 보호 코드 추가: overflow 방지
            n_left = len(good_left_inds)
            n_right = len(good_right_inds)

            if left_idx + n_left > max_pixels:
                print("⚠️ left_inds 배열 초과 발생! 일부 데이터는 무시됩니다.")
                n_left = max_pixels - left_idx
            if right_idx + n_right > max_pixels:
                print("⚠️ right_inds 배열 초과 발생! 일부 데이터는 무시됩니다.")
                n_right = max_pixels - right_idx

            left_inds[left_idx:left_idx+n_left] = good_left_inds[:n_left]
            right_inds[right_idx:right_idx+n_right] = good_right_inds[:n_right]
            left_idx += n_left
            right_idx += n_right

        # 슬라이스
        left_lane_inds = left_inds[:left_idx]
        right_lane_inds = right_inds[:right_idx]

        
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # 이상점 제거
        left_fit, right_fit = None, None
        if len(leftx) > 20:
            #polyfit : 주어진 x,y 데이터를 바탕으로 다항식을 구해주는 함수 여기서는 마지변수가 2여서 2차함수
            left_fit = np.polyfit(lefty, leftx, 2)
            leftx, lefty = self.remove_outliers(leftx, lefty, left_fit)
            if len(leftx) > 0:
                left_fit = np.polyfit(lefty, leftx, 2)
            else:
                left_fit = None
        if len(rightx) > 20:
            right_fit = np.polyfit(righty, rightx, 2)
            rightx, righty = self.remove_outliers(rightx, righty, right_fit)
            if len(rightx) > 0:
                right_fit = np.polyfit(righty, rightx, 2)
            else:
                right_fit = None


        # 시각화
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        
        ploty = np.linspace(0, height - 1, height)
        if left_fit is not None:
            left_fitx = np.polyval(left_fit, ploty)
            if draw:
                for i in range(len(ploty)-1):
                    cv.line(out_img, (int(left_fitx[i]), int(ploty[i])), (int(left_fitx[i+1]), int(ploty[i+1])), (255, 255, 0), 2)
        if right_fit is not None:
            right_fitx = np.polyval(right_fit, ploty)
            if draw:
                for i in range(len(ploty)-1):
                    cv.line(out_img, (int(right_fitx[i]), int(ploty[i])), (int(right_fitx[i+1]), int(ploty[i+1])), (0, 255, 255), 2)
        if left_fit is None or right_fit is None:
            return self.sliding_windows_visual(warped_img_ori, draw)
        else:
            if np.any(left_fitx >= right_fitx):
                return self.sliding_windows_visual(warped_img_ori, draw)
        return {
            "image": out_img,
            "left": {
                "fit": left_fit,
                "x": leftx,
                "y": lefty,
            },
            "right": {
                "fit": right_fit,
                "x": rightx,
                "y": righty,
            }
        }
    #위에있는 sliding window를 이용해 계산한 다항식을 기반으로 차선을 추적
    #처음 sliding window의 다항식을 쓰고 다음부터는 이 함수 스스로 계산한 다항식을 추적
    #속도가 빠른 대신 스스로 찾은 다항식을 추적하다보니 한번 엇나가면 복구가 힘들어 리셋 필요
    def quick_search(self, warped_img, draw):
        nonzero = warped_img.nonzero()
        nonzeroy = nonzero[0]
        nonzerox = nonzero[1]

        height = warped_img.shape[0]

        # 다항식 직접 계산으로 최적화
        y_vals = nonzeroy
        left_fitx = self.prev_left_fit[0]*y_vals**2 + self.prev_left_fit[1]*y_vals + self.prev_left_fit[2]
        right_fitx = self.prev_right_fit[0]*y_vals**2 + self.prev_right_fit[1]*y_vals + self.prev_right_fit[2]

        margin = self.margin
        left_lane_inds = (nonzerox > (left_fitx - margin)) & (nonzerox < (left_fitx + margin))
        right_lane_inds = (nonzerox > (right_fitx - margin)) & (nonzerox < (right_fitx + margin))

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        out_img = cv.merge([warped_img]*3) * 255  # np.dstack보다 약간 빠름

        left_fit, right_fit = None, None

        if len(leftx) > 20:
            left_fit = np.polyfit(lefty, leftx, 2)
            leftx, lefty = self.remove_outliers(leftx, lefty, left_fit)
            if len(leftx) > 0:
                left_fit = np.polyfit(lefty, leftx, 2)
            else:
                left_fit = None
        if len(rightx) > 20:
            right_fit = np.polyfit(righty, rightx, 2)
            rightx, righty = self.remove_outliers(rightx, righty, right_fit)
            if len(rightx) > 0:
                right_fit = np.polyfit(righty, rightx, 2)
            else:
                right_fit = None
            
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        if draw:
            ploty = np.linspace(0, height-1, height).astype(np.int32)
            if left_fit is not None:
                left_fitx = (left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]).astype(np.int32)
                pts = np.array([np.transpose(np.vstack([left_fitx, ploty]))], dtype=np.int32)
                cv.polylines(out_img, pts, isClosed=False, color=(0, 255, 255), thickness=2)
            if right_fit is not None:
                right_fitx = (right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]).astype(np.int32)
                pts = np.array([np.transpose(np.vstack([right_fitx, ploty]))], dtype=np.int32)
                cv.polylines(out_img, pts, isClosed=False, color=(255, 255, 0), thickness=2)

        return {
            "image": out_img,
            "left": {"fit": left_fit, "x": leftx, "y": lefty},
            "right": {"fit": right_fit, "x": rightx, "y": righty}
        }



    # Warp image perspective
#원근 변환
#img : 이미지, M : 원근변환을 위한 행렬
def warp(img, M):
    return cv.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv.INTER_NEAREST)

#원근변환을 위한 행렬 구하는 함수
#src : 원본 이미지에서 원근변환을 하고 싶은곳의 좌표값, dst : 원근 변환후의 좌표값
def warp_M(src=src, dst=dst):
    M = cv.getPerspectiveTransform(src, dst)
    return M

#역 원근변환을 위한 행렬 구하는 함수
#src : 원본 이미지에서 원근변환을 하고 싶은곳의 좌표값, dst : 원근 변환후의 좌표값
def Re_warp(src=src, dst=dst):
    Minv = cv.getPerspectiveTransform(dst, src)
    return Minv

# Crop image for region of interest
#영역 자르기
#img : 이미지, ROI : 자를 영역
def crop(img, ROI):
    # Create blank img with same size as input img
    blank = np.zeros(img.shape[:2], np.uint8)

    # Fill region of interest
    region_of_interest = cv.fillPoly(blank, ROI, 255)

    # Create image of interest with region (resize)
    return cv.bitwise_and(img, img, mask=region_of_interest)

# Merge to masks
#두 마스크에서 공통된 부분을 남기는 함수
#frame : 이미지, img1 : 마스크 1, img2 : 마스크 2
def merge(frame, img1, img2):
    both = frame.copy()
    both[np.where(np.logical_and(img1==0, img2==0))] = 0
    return both

#다항 곡선을 따라 선에 빈 공간 여부로 점선인지 실선으로 판단하는 함수
def detect_dash_line_along_curve(binary_img, fit, ploty, threshold_gap=50, threshold_segment=50):
    """
    다항 곡선을 따라 점선인지 실선인지 분석.
    - binary_img: 흑백 이미지 (차선만 흰색)
    - fit: np.polyfit으로 얻은 다항 계수
    - ploty: y좌표 배열
    - threshold_gap: 점선 판단 기준이 되는 최소 gap
    - threshold_segment: 실선 판단 기준이 되는 최소 선의 길이
    """
    gaps = []
    segments = []
    last_seen = None
    current_segment = 0
    
    for i, y in enumerate(ploty.astype(int)):
        x = int(np.polyval(fit, y))
        if 0 <= y < binary_img.shape[0] and 0 <= x < binary_img.shape[1]:
            radius = 20
            if np.any(binary_img[y, max(x - radius, 0): min(x + radius + 1, binary_img.shape[1])] > 0) or i == len(ploty) - 1:
                if last_seen is not None:
                    gap = y - last_seen
                    if gap > threshold_gap:
                        gaps.append(gap)
                        if current_segment > 20:
                            segments.append(current_segment)
                        current_segment = 0
                    else:
                        current_segment += gap
                last_seen = y

    if current_segment > 0:
        segments.append(current_segment)

    # 디버깅 출력
    """
    print("평균 gap:", np.mean(gaps) if gaps else 0)
    print("gap 표준편차:", np.std(gaps) if gaps else 0)
    print("gap 개수:", len(gaps))
    print("평균 segment 길이:", np.mean(segments) if segments else 0)
    print("segment 개수:", len(segments))
    """
    # 판단 조건
    if len(gaps) >= 2 and np.mean(gaps) > threshold_gap and len(segments) >= 2:
        return "dashed"
    else:
        return "solid"


#결과인 다항식을 바탕으로 선을 시각화
#지금은 사용 안함
#binary_img : 이진 이미지 데이터, fit : 다항식, ploty : y 범위, line_type : 선의 종류
def draw_lane_curve(binary_img, fit, ploty, line_type):
    color = (0, 255, 0) if line_type == "solid" else (0, 0, 255)
    curve_img = np.zeros((binary_img.shape[0], binary_img.shape[1], 3), dtype=np.uint8)
    x_vals = np.polyval(fit, ploty).astype(np.int32)
    for i in range(len(ploty)-1):
        pt1 = (x_vals[i], int(ploty[i]))
        pt2 = (x_vals[i+1], int(ploty[i+1]))
        if 0 <= pt1[0] < binary_img.shape[1] and 0 <= pt2[0] < binary_img.shape[1]:
            cv.line(curve_img, pt1, pt2, color, 3)
    return curve_img

#이미지에 투명도를 적용시켜 채우기 위한 함수
def blend_transparent_overlay(base_img, overlay_mask, color=(0, 255, 255), alpha=0.4):
    """
    base_img: 원본 BGR 이미지
    overlay_mask: 채워질 영역 (uint8 마스크, 255가 채워질 부분)
    color: 덧씌울 색상 (BGR)
    alpha: 투명도 (0: 완전 투명, 1: 불투명)
    """
    overlay = np.zeros_like(base_img, dtype=np.uint8)
    overlay[:] = color
    mask_3ch = cv.merge([overlay_mask] * 3)
    blended = cv.addWeighted(base_img, 1, cv.bitwise_and(overlay, mask_3ch), alpha, 0)
    return blended

#원근 변환이 된 이미지에서 진행된 차선 추적 결과값을 원본 이미지에 올리는 함수
#original_img : 원본 이미지, left_fit : 왼쪽 차선의 다항식, right_fit : 오른쪽 차선의 다항식, warped_shape : 원근 변환된 이미지의 shape,
#Minv : 역 원근변환을 위한 변환 행렬, left_type : 왼쪽 차선의 타입, right_type : 오른쪽 차선의 타입
#left_color : 왼쪽 차선의 결과 표시 색, right_color : 오른쪽 차선의 결과 표시 색, fill_color : 차선 사이 표시 색
def draw_lane_area_with_labels(original_img, left_fit, right_fit, warped_shape, Minv,
                               left_type="unknown", right_type="unknown",
                               left_color=(0, 255, 0), right_color=(255, 0, 0), fill_color=(0, 255, 255)):

    ploty = np.linspace(0, warped_shape[0] - 1, warped_shape[0])

    if left_fit is None or right_fit is None:
        return original_img  # early exit
    
    left_fitx = np.polyval(left_fit, ploty)
    left_pts = np.array([[left_fitx[i], ploty[i]] for i in range(len(ploty))], dtype=np.float32).reshape(-1, 1, 2)
    left_unwarped = cv.perspectiveTransform(left_pts, Minv)

    right_fitx = np.polyval(right_fit, ploty)
    right_pts = np.array([[right_fitx[i], ploty[i]] for i in range(len(ploty))], dtype=np.float32).reshape(-1, 1, 2)
    right_unwarped = cv.perspectiveTransform(right_pts, Minv)
    # 역투영
    
    

    # 차선 영역 그리기

    

    result = original_img.copy()

    lane_poly = np.vstack((left_unwarped, np.flipud(right_unwarped)))
#cv.fillPoly(result, [np.int32(lane_poly)], fill_color)

    lane_mask = np.zeros(original_img.shape[:2], dtype=np.uint8)
    cv.fillPoly(lane_mask, [np.int32(lane_poly)], 255)
    result = blend_transparent_overlay(original_img, lane_mask, color=fill_color, alpha=0.4)

    # 좌우 선 그리기
    cv.polylines(result, [np.int32(left_unwarped)], False, left_color, 3)
    cv.polylines(result, [np.int32(right_unwarped)], False, right_color, 3)
    
# 박스 그리기: 좌우 차선 경계 사각형
    def draw_label_box(unwarped_pts, color, label):
        xs = [pt[0][0] for pt in unwarped_pts]
        ys = [pt[0][1] for pt in unwarped_pts]
        x_min, x_max = int(min(xs)), int(max(xs))
        y_min, y_max = int(min(ys)), int(max(ys))
        cv.rectangle(result, (x_min, y_min), (x_max, y_max), color, 2)
        cv.putText(result, label, (x_min, y_min - 10),
                    cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv.LINE_AA)

# 박스 + 라벨 추가
    draw_label_box(left_unwarped, left_color, f"Left: {left_type}")
    draw_label_box(right_unwarped, right_color, f"Right: {right_type}")

    return result

#lab 방식으로 clahe 동작
#명암 대비를 올려서 어두운곳도 잘 보이게
#cpu를 많이 먹는다고하여 제외됨
#img : 이미지
def clahe(img):

    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv.merge((l_clahe, a, b))
    img_clahe = cv.cvtColor(lab_clahe, cv.COLOR_LAB2BGR)

    return img_clahe

#hls 방식으로 clahe 동작
#명암 대비를 올려서 어두운곳도 잘 보이게
#cpu를 많이 먹는다고하여 제외됨
#img : 이미지
def hls_clahe(img):
    hls = cv.cvtColor(img, cv.COLOR_BGR2HLS)
    h, l, s = cv.split(hls)

    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)

    hls_eq = cv.merge((h, l_eq, s))
    img_clahe = cv.cvtColor(hls_eq, cv.COLOR_HLS2BGR)
    return img_clahe

#이미지의 특정 탐지 영역의 밝기 확인
#image : 이미지, x_ratio : 탐지 영역의 x 중앙값, y_ratio : 탐지 영역의 y 상단 값, region_size : 탐지 영역의 크기 
def get_region_brightness(image, x_ratio=0.5, y_ratio=0.6, region_size=0.2):
    height, width = image.shape[:2]
    x_start = int(width * (x_ratio - region_size / 2))
    x_end = int(width * (x_ratio + region_size / 2))
    y_start = int(height * y_ratio)
    y_end = int(height * (y_ratio + region_size))

    region = image[y_start:y_end, x_start:x_end]


    brightness = np.mean(region)
    return brightness

# 호출
#color 방식으로 차선 탐지
#전처리 과정이 color 방식으로 다를 뿐 그 이후는 같음
#frame : 이미지, M : 원근변환을 위한 행렬, Minv : 역 원근변환을 위한 행렬, LT : 차선감지 클래스
def line_check(frame, M, Minv, LT):
    orig = frame.copy()
    """
    img_clahe = hls_clahe(orig)

    color = color_space_hls(img_clahe)
    brightness = get_region_brightness(img_clahe)
    """

    #hls 값을 통해 흰색과 노란색 계통만 남기고 흑백화
    color = color_space_hls(orig)

    #원본 이미지의 밝기 평균값 확인
    brightness = get_region_brightness(orig)
    #해당 밝기 평균값을 바탕으로 80 ~ 240 사이의 threshold 값 구하기
    threshold_val = int(np.clip(brightness * 1.2, 80, 240))
    #해당 threshold를 바탕으로 흑백 이미지에서 2진 이미지로 변경
    _, binary_result = cv.threshold(color, threshold_val, 255, cv.THRESH_BINARY)


    #return cv.bitwise_and(binary_result, binary_result, mask=shadow_mask)

    #여기서 부터는 동일
    #차선 판단을 수월하게 하기 위한 원근변환
    color = warp(binary_result, M)



    # Step 3: Sliding windows to get curve points    
    #midpoint, lefts, rights = sliding_windows(color)

    
    #result = central_sliding_windows_based_on_existing(color, nwindows= 5, minimum =100, draw=True)
    #전처리 후 차선 감지
    result = LT.update(color)
    
    
    #여기서부터는 감지된 차선을 바탕으로 차선의 종류(실선, 점선) 판단


    #차선 데이터를 바탕으로 왼쪽과 오른쪽 차선을 색으로 분리
    lower_red = np.array([0, 0, 200])
    upper_red = np.array([50, 50, 255])
    #mask_red = cv.inRange(result["image"], lower_red, upper_red)

    # 파란색 마스크
    lower_blue = np.array([200, 0, 0])
    upper_blue = np.array([255, 50, 50])
    #mask_blue = cv.inRange(result["image"], lower_blue, upper_blue)

    # 두 마스크 합치기
    
    mask_combined = cv.inRange(result["image"], lower_red, upper_red)
    mask_combined |= cv.inRange(result["image"], lower_blue, upper_blue)


    ploty = np.linspace(0, result["image"].shape[0] - 1, num=result["image"].shape[0])
    if result["left"]["fit"] is not None:
        #차선의 다항식이 있으면 그 수식을 바탕으로 차선 종류 판단
        #다항식을 이미지에서 따라가며 중간에 빈 공간이 있나 여부로 점선, 실선 판단
        left_line_type = detect_dash_line_along_curve(mask_combined, result["left"]["fit"], ploty)
        #left_lane_img = draw_lane_curve(mask_combined, result["left"]["fit"], ploty, left_line_type)
    else:
        left_line_type = "unknown"

    
    if result["right"]["fit"] is not None:
        right_line_type = detect_dash_line_along_curve(mask_combined, result["right"]["fit"], ploty)
        #right_lane_img = draw_lane_curve(mask_combined, result["right"]["fit"], ploty, right_line_type)
    else:
        right_line_type = "unknown"
        
    #결과를 원본 이미지에 표시하기
    #원근 변환된 이미지를 원본에 맞춰서 역 원근변환 
    result = draw_lane_area_with_labels(
        #original_img=cv.cvtColor(orig, cv.COLOR_BGR2RGB),
        original_img=orig,
        left_fit=result["left"]["fit"],
        right_fit=result["right"]["fit"],
        warped_shape=color.shape,
        Minv=Minv,
        left_color=(0, 255, 0),     # 초록
        right_color=(255, 0, 0),    # 파랑
        fill_color=(0, 255, 255),    # 차선 사이 채우기 (노랑)
        left_type=left_line_type,
        right_type=right_line_type
    )
    return result

#소벨 에지를 통해 2진 데이터를 내보내는 함수
#img : 원본 이미지
def combined_threshold(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 소벨 에지 검출
    #밝기 값이 급격히 변하는 영역을 감지 -> 윤곽선이 감지가 됨
    sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    #cv.imshow('scaled_sobel1', scaled_sobel)
    # 임계값 적용
    #임계값으로 마스킹 해서 2진 이미지로 변환
    thresh_min = 15
    thresh_max = 100
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    """
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(sxbinary == 1)] = 255
    cv.imshow('sxbinary', combined_binary)
    """
    # 색상 임계값
    #외곽만 하면 안되는 경우 있어서 어느정도 색상도 약하게 마스킹을 해서 추출
    hls = cv.cvtColor(img, cv.COLOR_BGR2HLS)
    s_channel = hls[:, :, 2]
    s_thresh_min = 100
    s_thresh_max = 150
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
    """
    combined_binary = np.zeros_like(s_binary)
    combined_binary[(s_binary == 1)] = 255
    cv.imshow('s_binary', combined_binary)
    """
    # 결합
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(sxbinary == 1) | (s_binary == 1)] = 255
    return combined_binary

def open_img(img, iterations):
    return cv.morphologyEx(img, cv.MORPH_OPEN, kernel_small, iterations=iterations)

# 최초호출 
#soble 방식으로 차선 탐지
#전처리 과정이 soble 방식으로 다를 뿐 그 이후는 같음
def line_check_sobel(frame, M, Minv, LT):
    orig = frame.copy()

    
    """
    img_clahe = hls_clahe(orig)

    blurred = cv.GaussianBlur(img_clahe, (5, 5), 0)

    sobel_test = combined_threshold(blurred)
    """
    #sobel 방식을 통해 윤곽선을 바탕으로 2진화(color 방식을 약하게 적용해 선 내부도 어느정도 강조)
    sobel_test = combined_threshold(orig)
    #노이즈 제거를 위환 open 연산
    binary_result = open_img(sobel_test, 1)

    #여기서부터는 동일 line_check 에 주석 하겠음
    color = warp(binary_result, M)



    # Step 3: Sliding windows to get curve points    
    #midpoint, lefts, rights = sliding_windows(color)
    
    result = LT.update(color)

    
    lower_red = np.array([0, 0, 200])
    upper_red = np.array([50, 50, 255])
    #mask_red = cv.inRange(result["image"], lower_red, upper_red)

    # 파란색 마스크
    lower_blue = np.array([200, 0, 0])
    upper_blue = np.array([255, 50, 50])
    #mask_blue = cv.inRange(result["image"], lower_blue, upper_blue)

    # 두 마스크 합치기
    
    mask_combined = cv.inRange(result["image"], lower_red, upper_red)
    mask_combined |= cv.inRange(result["image"], lower_blue, upper_blue)


    ploty = np.linspace(0, result["image"].shape[0] - 1, num=result["image"].shape[0])
    if result["left"]["fit"] is not None:
        left_line_type = detect_dash_line_along_curve(mask_combined, result["left"]["fit"], ploty)
        #left_lane_img = draw_lane_curve(mask_combined, result["left"]["fit"], ploty, left_line_type)
    else:
        left_line_type = "unknown"

    
    if result["right"]["fit"] is not None:
        right_line_type = detect_dash_line_along_curve(mask_combined, result["right"]["fit"], ploty)
        #right_lane_img = draw_lane_curve(mask_combined, result["right"]["fit"], ploty, right_line_type)
    else:
        right_line_type = "unknown"

    result = draw_lane_area_with_labels(
        #original_img=cv.cvtColor(orig, cv.COLOR_BGR2RGB),
        original_img=orig,
        left_fit=result["left"]["fit"],
        right_fit=result["right"]["fit"],
        warped_shape=color.shape,
        Minv=Minv,
        left_color=(0, 255, 0),     # 초록
        right_color=(255, 0, 0),    # 파랑
        fill_color=(0, 255, 255),    # 차선 사이 채우기 (노랑)
        left_type=left_line_type,
        right_type=right_line_type
    )
    return result


# Open video file


#여기서부터 주요 예시
#video_file = 'project'

def example():
    import os
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge
    import threading
    
    # ROS2 초기화 (도메인 ID 10 사용)
    os.environ['ROS_DOMAIN_ID'] = '10'
    rclpy.init()
    
    class VideoSubscriber(Node):
        def __init__(self, topic_name: str = 'my_camera/node'):
            super().__init__('line_check_subscriber')
            
            self.bridge = CvBridge()
            self.current_frame = None
            self.frame_lock = threading.Lock()
            
            self.subscription = self.create_subscription(
                Image,
                topic_name,
                self.image_callback,
                10
            )
            
            self.get_logger().info(f'Line Check subscriber started, listening to: {topic_name}')
        
        def image_callback(self, msg):
            try:
                # ROS2 Image 메시지를 OpenCV 프레임으로 변환
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                with self.frame_lock:
                    self.current_frame = cv_image.copy()
            except Exception as e:
                self.get_logger().error(f"Error converting ROS2 image: {str(e)}")
        
        def get_latest_frame(self):
            with self.frame_lock:
                return self.current_frame.copy() if self.current_frame is not None else None
    
    # ROS2 subscriber 생성 
    video_subscriber = VideoSubscriber('my_camera/node')
    
    LT = LaneTracker(margin=50)

    # 첫 프레임을 받을 때까지 대기
    print("Waiting for first frame from ROS2 topic...")
    timeout_count = 0
    while timeout_count < 50:  # 최대 5초 대기
        rclpy.spin_once(video_subscriber, timeout_sec=0.1)
        frame = video_subscriber.get_latest_frame()
        if frame is not None:
            break
        timeout_count += 1
    
    if frame is None:
        print("Error: Could not receive frames from ROS2 topic")
        video_subscriber.destroy_node()
        rclpy.shutdown()
        return

    # If challenge video is played -> Define different points for transformation 

    #차선 인식을 위한 다각형 좌표
    height, width = frame.shape[:2]
    src = np.float32([
        [width * 0.45, height * 0.57],
        [width * 0.55, height * 0.57],
        [width * 0.9, height],
        [width * 0.1, height]
    ])
    dst = np.float32([
        [width * 0.3, 0],
        [width * 0.7, 0],
        [width * 0.7, height],
        [width * 0.3, height]
    ])
    M = warp_M(src, dst)
    Minv = Re_warp(src, dst)
    
    # 다각형 좌표 시각화 함수 추가
    def draw_polygon_coordinates(img, src_points, dst_points, show_both=True):
        """다각형 좌표를 시각화하는 함수"""
        img_vis = img.copy()
        
        if show_both:
            # 원본 이미지에 src 포인트 그리기 (녹색)
            src_polygon = np.array(src_points, dtype=np.int32)
            cv2.polylines(img_vis, [src_polygon], True, (0, 255, 0), 3)
            for i, point in enumerate(src_points):
                cv2.circle(img_vis, (int(point[0]), int(point[1])), 8, (0, 255, 0), -1)
                cv2.putText(img_vis, f'S{i}', (int(point[0])-10, int(point[1])-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return img_vis
    
    # 첫 번째 프레임에서 다각형 좌표 상세 시각화
    frame_with_polygons = draw_polygon_coordinates(frame, src, dst)
    
    # 좌표 정보를 추가로 표시
    cv2.putText(frame_with_polygons, 'Source Polygon (Green)', (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 각 좌표점 정보 출력
    for i, point in enumerate(src):
        text = f'S{i+1}: ({int(point[0])}, {int(point[1])})'
        cv2.putText(frame_with_polygons, text, (10, 60 + i*25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.imshow('Polygon Coordinates Visualization', frame_with_polygons)
    
    # 콘솔에 상세 정보 출력
    print("=== 차선 인식 다각형 좌표 ===")
    print("Source Points (원본 이미지 좌표):")
    for i, (x, y) in enumerate(src):
        print(f"  S{i+1}: ({x:.1f}, {y:.1f}) -> ({x/width*100:.1f}% width, {y/height*100:.1f}% height)")
    
    print("\nDestination Points (변환 후 좌표):")
    for i, (x, y) in enumerate(dst):
        print(f"  D{i+1}: ({x:.1f}, {y:.1f}) -> ({x/width*100:.1f}% width, {y/height*100:.1f}% height)")
    
    print("\n창에서 아무 키나 누르면 차선 인식을 시작합니다...")
    cv2.waitKey(0)  # 키를 누를 때까지 대기
    cv2.destroyWindow('Polygon Coordinates Visualization')

    prev_time = 0

    show_polygon_overlay = False
    print("Line detection started. Press 'q' to quit, 'p' to toggle polygon overlay...")

    while rclpy.ok():
        rclpy.spin_once(video_subscriber, timeout_sec=0.1)
        
        frame = video_subscriber.get_latest_frame()
        if (frame is None):
            continue
        # 다각형 오버레이를 표시할지
        if show_polygon_overlay:
            frame_with_polygons = draw_polygon_coordinates(frame, src, dst)
            cv2.putText(frame_with_polygons, "POLYGON OVERLAY ON", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            frame = line_check(frame_with_polygons, M, Minv, LT)
        else:
            frame = line_check(frame, M, Minv, LT)

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        #frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        cv.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv.imshow('Line Detection', frame)

        # UI에 정보 추가
        if show_polygon_overlay:
            cv.putText(frame, "Polygon Overlay ON", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv.imshow('Line Detection', frame)

        key = cv.waitKey(25) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            show_polygon_overlay = not show_polygon_overlay
            print(f"Polygon overlay: {'ON' if show_polygon_overlay else 'OFF'}")

    # 정리
    try:
        video_subscriber.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    except:
        pass
    finally:
        cv.destroyAllWindows()
if __name__ == "__main__":
    example()